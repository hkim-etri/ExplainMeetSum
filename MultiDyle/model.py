import gc
import json
import os
import random
import shutil

import numpy as np
import torch
import torch.multiprocessing

from typing import List
from typing import Tuple

from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

from config import Config
from MultiDyle.dyle.dynamic_rag import DynamicRagForGeneration
from MultiDyle.preprocess import QMSumSent
from transformers import (RobertaTokenizer,
                          RobertaForTokenClassification,
                          BartTokenizer,
                          AdamW)
from MultiDyle.dyle.clean_utils import tokenize
from MultiDyle.dyle.utils import gpu_wrapper, rouge_with_pyrouge


config = Config()
torch.multiprocessing.set_sharing_strategy('file_system')

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
if config.gpu:
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class ExperimentMultiDyle():
    def __init__(self, load_train=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever_list = [f'retriever{i}' for i in range(1, config.num_retriever + 1)]
        self.modules = self.retriever_list + ['generator', 'criterion_cls']

        # Load retriever tokenizer.
        self.retriever_tokenizer = RobertaTokenizer.from_pretrained(
            config.retriever_name_or_path
        )
        # Load retriever model.
        for retriever in self.retriever_list:
            setattr(self, retriever, RobertaForTokenClassification.from_pretrained(
                config.retriever_name_or_path,
                num_labels=1,
                gradient_checkpointing=True
            ))
        # Load generator tokenizer.
        self.generator_tokenizer = BartTokenizer.from_pretrained(
            config.generator_name_or_path
        )
        # Load generator model.
        self.generator = DynamicRagForGeneration.from_pretrained(
            config.generator_name_or_path,
            n_docs=sum(config.extractor_top_k),
            gradient_checkpointing=True
        )

        # Load loss.
        self.criterion_cls = torch.nn.CrossEntropyLoss(reduction='none')

        self.eval_model_dir = config.eval_model_dir

        if load_train:
            load_module_list = ['generator']
            print(f"using pretrained models : {load_module_list}")
            
            for module in self.modules:
                # only load `generator` checkpoint
                if module in load_module_list:
                    path = f'{config.dyle_ckpt_dir}/best-{module}.ckpt'
                    getattr(self, module).load_state_dict(
                        torch.load(path, map_location=lambda storage, loc: storage)
                    )

        # Load dataset.
        print('----- Loading data -----')

        if load_train:
            self.train_set = QMSumSent(
                'train',
                retriever_tokenizer=self.retriever_tokenizer,
                generator_tokenizer=self.generator_tokenizer
            )
            self.val_set = QMSumSent(
                'val',
                retriever_tokenizer=self.retriever_tokenizer,
                generator_tokenizer=self.generator_tokenizer
            )
        # else:
        self.test_set = QMSumSent(
            'test',
            retriever_tokenizer=self.retriever_tokenizer,
            generator_tokenizer=self.generator_tokenizer
        )

        for module in self.modules:
            print('--- {}: '.format(module))
            setattr(self, module, gpu_wrapper(getattr(self, module)))

        self.scopes = {'cls': self.retriever_list, 'gen': ['generator']}
        for scope in self.scopes.keys():
            setattr(self, scope + '_lr', getattr(config, scope + '_lr'))

        self.top_k_ckpt: List[dict] = []

    def restore_model(self, modules, dir):
        assert dir != None
        print('Loading the trained best models...')

        for module in modules:
            path = os.path.join(dir, '{}.ckpt'.format(module))
            print(path)
            getattr(self, module).load_state_dict(
                torch.load(path, map_location=lambda storage, loc: storage),
                strict=True
            )

    def zero_grad(self):
        for scope in self.scopes:
            getattr(self, scope + '_optim').zero_grad()

    def step(self, scopes):
        if config.max_grad_norm is not None:
            grouped_params = []
            for scope in scopes:
                grouped_params.extend(getattr(self, scope + '_grouped_parameters'))

            clip_grad_norm_(grouped_params, config.max_grad_norm)

        for scope in scopes:
            # Optimize.
            getattr(self, scope + '_optim').step()

    def set_training(self, mode):
        for module in self.modules:
            getattr(self, module).train(mode=mode)

    def train(self):
        self.build_optim()
        for epoch in range(config.num_epochs):
            print("Training Epoch # : {}".format(epoch))
            self.train_epoch()
            self.eval_tmp(epoch_id=epoch)
    
    def test(self):
        self.restore_model(self.modules[:-1], self.eval_model_dir)
        self.eval(is_train=False)

    def build_optim(self):
        # Set trainable parameters, according to the frozen parameter list.
        for scope in self.scopes.keys():
            optimizer_grouped_parameters = [
                {'params': [],
                 'weight_decay': config.weight_decay},
                {'params': [],
                 'weight_decay': 0.0},
            ]
            no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']

            for module in self.scopes[scope]:
                for n, p in getattr(self, module).named_parameters():
                    if p.requires_grad:
                        # Weight decay.
                        if not any(nd in n for nd in no_decay):
                            print("[{} Trainable:]".format(module), n)
                            optimizer_grouped_parameters[0]['params'].append(p)
                        else:
                            print("[{} Trainable (bias/LN):]".format(module), n)
                            optimizer_grouped_parameters[1]['params'].append(p)
                    else:
                        print("[{} Frozen:]".format(module), n)

            # Adam optimizer, scope will be 'cls' or 'gen'
            setattr(
                self,
                scope + '_optim',
                AdamW(optimizer_grouped_parameters, lr=getattr(self, scope + '_lr'))
            )
            setattr(
                self,
                scope + '_grouped_parameters',
                optimizer_grouped_parameters[0]['params'] + optimizer_grouped_parameters[1]['params']
            )
    
    def get_retriever_cls_logits_list(self, input_ids, cls_ids):
        logit_list = []

        for retriever in self.retriever_list:
            _retriever = getattr(self, retriever)
            retriever_outputs = _retriever(
                input_ids=input_ids.squeeze(0).to(self.device),
                output_hidden_states=True,
            )
            logits = retriever_outputs.logits.squeeze(2).contiguous().view(-1)
            cls_logits = logits[cls_ids.squeeze(0).cpu().tolist()].unsqueeze(0)
            logit_list.append(cls_logits)
            
        return logit_list
    
    def get_retriever_loss(self, cls_logits_list, oracle_list):
        retriever_loss = 0
        for cls_logits, oracle in zip(cls_logits_list, oracle_list):
            oracle = unpad_list(oracle)
            # no oracle
            if len(oracle) == 0:
                continue
            retriever_loss += sum([
                self.criterion_cls(
                    input=cls_logits,
                    target=gpu_wrapper(torch.LongTensor([turn_id]))
                ) / len(oracle)
                for turn_id in oracle.cpu().tolist()
            ])
        return retriever_loss
    
    def get_generator_loss(
        self,
        retriever_cls_logit_list,
        oracle_list,
        context_input_ids,
        context_attention_mask,
        labels,
    ):
        merged_doc_scores, merged_top_k_indices = self.get_top_k_doc_scores(
            retriever_cls_logit_list,
            oracle_list,
        )
        context_input_ids = context_input_ids[:, merged_top_k_indices].contiguous().view(
            context_input_ids.shape[0] * sum(config.extractor_top_k),
            -1,
        )
        context_attention_mask = context_attention_mask[:, merged_top_k_indices].contiguous().view(
            context_attention_mask.shape[0] * sum(config.extractor_top_k),
            -1
        )
        generator_outputs = self.generator(
            context_input_ids=context_input_ids.to(self.device),
            context_attention_mask=context_attention_mask.to(self.device),
            doc_scores=merged_doc_scores.to(self.device),
            labels=labels.to(self.device)
        )
        seq_loss = generator_outputs.loss
        consistency_loss = generator_outputs.consistency_loss
        
        return seq_loss, consistency_loss
        
    
    def extract_top_k_sentences(self, cls_logits, oracle, k: int, is_train: bool):
        # Number of sentences to be extracted
        top_k = min(cls_logits.shape[1], k)
        
        # Unpad oracle
        oracle = unpad_list(oracle)

        # Extract top k cls_logits in descending order
        retriever_doc_scores, retriever_topk_indices = torch.topk(
            cls_logits,
            k=top_k,
            dim=1
        )
        retriever_topk_indices = retriever_topk_indices[0].cpu().tolist()

        if is_train:
            # If oracle exists
            if len(oracle) != 0:
                oracle_top_k_indices = oracle.cpu().tolist()[:top_k]
                
                # Use oracle & extracted ones
                if config.hybrid_train and len(oracle_top_k_indices) < top_k:
                    oracle_top_k_indices.extend(retriever_topk_indices)
                    retriever_topk_indices = list(set(oracle_top_k_indices))[:top_k]
                # Use only oracles
                else:
                    retriever_topk_indices = oracle_top_k_indices

                retriever_doc_scores = cls_logits[:, retriever_topk_indices]

        # Number of indices is lesser than pre-defined ones
        if len(retriever_topk_indices) < k:
            # fill paddings
            retriever_doc_scores = torch.cat(
                [
                    retriever_doc_scores,
                    gpu_wrapper(torch.zeros((1, k - len(retriever_topk_indices)))).fill_(-float('inf'))
                ],
                dim=1
            )
            retriever_topk_indices = retriever_topk_indices + \
                [retriever_topk_indices[-1]] * (k - len(retriever_topk_indices))
        
        return retriever_doc_scores, retriever_topk_indices

    def get_top_k_doc_scores(
        self,
        cls_logits_list,
        oracle_list,
        is_train: bool= True
    ):
        """Return merged top k doc_scores with their indices
        """
        doc_scores_list = list()
        merged_topk_indices = list()

        for cls_logits, oracle, top_k in zip(cls_logits_list, oracle_list, config.extractor_top_k):
            retriever_doc_scores, retriever_topk_indices = self.extract_top_k_sentences(
                cls_logits, oracle, top_k, is_train
            )
            doc_scores_list.append(retriever_doc_scores)
            merged_topk_indices.extend(retriever_topk_indices)
        
        merged_doc_scores = torch.cat(doc_scores_list, dim=1)

        return merged_doc_scores, merged_topk_indices

    def train_epoch(self):
        torch.cuda.empty_cache()
        gc.collect()
        
        self.set_training(mode=True)
        train_dataloader = DataLoader(
            self.train_set,
            batch_size=config.train_batch_size // config.gradient_accumulation_steps,
            shuffle=True,
            num_workers=config.num_workers
        )
        for data_idx, data in enumerate(tqdm(train_dataloader), start=1):
            data = self.cuda_data(*data)
            
            retriever_input_ids, retriever_cls_ids, retriever_oracle_list, \
            context_input_ids, context_attention_mask, labels = data
            
            retriever_oracle_list = retriever_oracle_list.squeeze()
            
            # Forward.
            retriever_cls_logit_list: list = self.get_retriever_cls_logits_list(
                retriever_input_ids,
                retriever_cls_ids
            )
            
            # Oracle loss with cross entopy
            ret_loss = self.get_retriever_loss(
                retriever_cls_logit_list,
                retriever_oracle_list
            )

            # Generation loss.
            seq_loss, consistency_loss = self.get_generator_loss(
                retriever_cls_logit_list,
                retriever_oracle_list,
                context_input_ids,
                context_attention_mask,
                labels,
            )

            tot_loss = seq_loss * config.loss_alpha + ret_loss \
                    + config.consistency_alpha * consistency_loss

            # Backward.
            if config.gradient_accumulation_steps > 1:
                tot_loss = tot_loss / config.gradient_accumulation_steps

            tot_loss.backward()

            if data_idx % config.gradient_accumulation_steps == 0:
                self.step(['cls', 'gen'])
                self.zero_grad()
    
    # TODO: remove after test done!
    def eval_tmp(self, epoch_id):
        self.eval(is_train=True, epoch_id=epoch_id)
        self.eval(is_train=False)
        
    def eval(self, is_train: bool, epoch_id=0):
        torch.cuda.empty_cache()
        gc.collect()

        self.set_training(mode=False)
        set_name = "val" if is_train else "test"
        dataset = getattr(self, set_name + '_set')

        print("\n\n\n\n***** Running evaluation *****")
        print(f'beam_size = {config.beam_size}')
        print(f'{set_name} dataset')

        predictions = self.run_eval(dataset, config.beam_size)
        eval_result: Tuple[float, float, float] = rouge_with_pyrouge(
            preds=predictions,
            refs=dataset.eval_references
        )
        # average rouge score
        avg_score = sum(eval_result) / 3
        
        print(eval_result)
        print(f"Reference-all : {len(dataset.eval_references)}")
        print("-------------------------")

        # validation. Save model
        if is_train:
            if len(self.top_k_ckpt) == 0 or avg_score > self.top_k_ckpt[-1]['score']:
                self.save_all_bests(epoch_id, avg_score)
        print('\n\n')

    # TODO: will be replaced..
    def save_all_bests(self, epoch_id, score):
        ckpt_name = f'epochs_{epoch_id}--val_{score:.4f}'

        ckpt_path = os.path.join(config.save_model_dir, ckpt_name)
        os.makedirs(ckpt_path, exist_ok=True)

        # Retrievers, Generator
        for module in self.modules[:-1]:
            path = os.path.join(ckpt_path, f'{module}.ckpt')
            torch.save(getattr(self, module).state_dict(), path)

        print('Saved model checkpoints into {}...\n\n\n'.format(ckpt_path))

        self.top_k_ckpt.append({'name': ckpt_name, 'score': score})
        self.top_k_ckpt.sort(key=lambda x: x['score'], reverse=True)
        remove_ckpt = self.top_k_ckpt[config.save_top_k:]
        self.top_k_ckpt = self.top_k_ckpt[:config.save_top_k]

        print(f'top{len(self.top_k_ckpt)} checkpoints')
        for ckpt_idx, ckpt in enumerate(self.top_k_ckpt):
            # print model info
            print(" -{} : {}".format(ckpt_idx, ckpt['name']))
        print('\n\n')
        
        for ckpt in remove_ckpt:
            shutil.rmtree(os.path.join(config.save_model_dir, ckpt['name']))

    def run_eval(self, dataset, beam_size):
        rouge_1_values = list()
        rouge_2_values = list()
        rouge_l_values = list()

        predictions = list()

        print(f'  Num examples = {len(dataset)}')
        eval_dataloader = DataLoader(
            dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )

        for data_idx, data in enumerate(tqdm(eval_dataloader)):
            data = self.cuda_data(*data)

            retriever_input_ids, retriever_cls_ids, retriever_oracle_list, \
            context_input_ids, context_attention_mask, _ = data
            
            retriever_oracle_list = retriever_oracle_list.squeeze()

            # Forward (prediction).
            with torch.no_grad():
                retriever_cls_logit_list: list = self.get_retriever_cls_logits_list(
                    retriever_input_ids,
                    retriever_cls_ids
                )
                merged_doc_scores, merged_top_k_indices = self.get_top_k_doc_scores(
                    retriever_cls_logit_list,
                    retriever_oracle_list,
                    is_train = False
                )
                print(merged_top_k_indices, '\n')

                context_input_ids = context_input_ids[:, merged_top_k_indices].contiguous().view(
                    context_input_ids.shape[0] * sum(config.extractor_top_k),
                    -1,
                )
                context_attention_mask = context_attention_mask[:, merged_top_k_indices].contiguous().view(
                    context_attention_mask.shape[0] * sum(config.extractor_top_k),
                    -1
                )
                outputs = self.generator.generate(
                    context_input_ids=context_input_ids,
                    context_attention_mask=context_attention_mask,
                    doc_scores=merged_doc_scores,
                    num_beams=beam_size,
                    min_length=config.min_length,
                    max_length=config.max_target_len,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                    length_penalty=config.length_penalty,
                )

                # Predictions.
                decoded_pred = self.generator_tokenizer.batch_decode(outputs, skip_special_tokens=True)

                cleaned_prediction = ["\n".join(sent_tokenize(tokenize(pred))) for pred in decoded_pred]
                predictions.extend(cleaned_prediction)

                r_text = [dataset.eval_references[data_idx]]
                p_text = cleaned_prediction

                print(f" - Reference  : {r_text}")
                print(f" - Prediction : {p_text}")
                rouge1, rouge2, rougeL = rouge_with_pyrouge(preds=p_text, refs=r_text)

                rouge_1_values.append(rouge1)
                rouge_2_values.append(rouge2)
                rouge_l_values.append(rougeL)

                print(f" - ROUGE : {rouge1}, {rouge2}, {rougeL}\n\n")

        return predictions

    def number_parameters(self):
        print('Number of retriever parameters', sum(p.numel() for p in self.retriever.parameters()))
        print('Number of retriever parameters', sum(p.numel() for p in self.retriever2.parameters()))
        print('Number of generator parameters', sum(p.numel() for p in self.generator.parameters()))

    @staticmethod
    def cuda_data(*data, **kwargs):
        if len(data) == 0:
            raise ValueError()
        elif len(data) == 1:
            return gpu_wrapper(data[0], **kwargs)
        else:
            return [gpu_wrapper(item, **kwargs) for item in data]


def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = parameters if isinstance(parameters, list) else [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm

def unpad_list(input_list: List[torch.LongTensor], pad_value: int = -1):
    """Unpad oracle
    """
    input_list = input_list.cpu().tolist()
    input_list = input_list if isinstance (input_list, list) else [input_list]
    
    while input_list[-1] == pad_value:
        input_list = input_list[:-1]
        if not input_list:
            break
        
    return torch.LongTensor(input_list)
