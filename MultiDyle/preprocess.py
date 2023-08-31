import glob
import json
import os

import torch

from typing import List, Dict

from torch.utils import data
from nltk import sent_tokenize, word_tokenize
from rouge import Rouge
from tqdm import tqdm

from config import Config
from MultiDyle.dyle.clean_utils import clean_data, tokenize


config = Config()


def rouge(dec, ref):
    if dec == '' or ref == '':
        return 0.0
    rouge = Rouge()
    scores = rouge.get_scores(dec, ref)
    return (scores[0]['rouge-1']['f'] + scores[0]['rouge-2']['f'] + scores[0]['rouge-l']['f']) / 3

def pad_list(input_list: list, pad_value: int = -1):
    """Pad list to convert tensor
    """
    max_len = max(list(map(len, input_list)))
    for idx in range(len(input_list)):
        input_list[idx].extend([pad_value] * (max_len - len(input_list[idx])))


class QMSumSent(data.Dataset):
    """The QMSum dataset."""

    def __init__(self, mode, retriever_tokenizer, generator_tokenizer):
        super().__init__()
        
        self.mode = mode
        self.retriever_tokenizer = retriever_tokenizer
        self.generator_tokenizer = generator_tokenizer

        self.root = config.dataset
        self.features = list()
        self.cached_dataset = f"{self.root}/{config.cached_dataset}_{self.mode}"
        self.oracle_path_list: list = config.oracle_path_list

        file_names = glob.glob(f'{self.root}/{mode}/*.json')
        # get sorted file names
        self.file_names = sorted(list(map(lambda x: x.split('/')[-1].split('.')[0] ,file_names)))

        for f in self.file_names:
            print(f)

        self.get_references()
        self.load_features_from_cache()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        retriever_inputs, generator_inputs = [
            {k: torch.LongTensor(v) for k, v in inputs.items()}
            for inputs in self.features[index]
        ]
        
        return (
            retriever_inputs['input_ids'],
            retriever_inputs['cls_ids'],
            retriever_inputs['oracle'], # List of oracles
            generator_inputs['context_input_ids'],
            generator_inputs['context_attention_mask'],
            generator_inputs['labels'],
        )

    def get_references(self):
        self.eval_references = []
        self.eval_file_query_names = []

        for file_name in self.file_names:
            file_path = f"{self.root}/{self.mode}/{file_name}.json"
            
            with open(file_path) as f:
                session = json.load(f)
                
                for query_list in ['general_query_list', 'specific_query_list']:
                    for idx, pair in enumerate(session['explainable_qmsum'][query_list]):
                        eval_reference = "\n".join(sent_tokenize(' '.join(word_tokenize(pair['answer'].lower()))))

                        self.eval_references.append(eval_reference)
                        self.eval_file_query_names.append({"file_name":file_name, "query_name":"generalB{}".format(idx), "query_type":"general", "query_i":idx})    

        assert len(self.eval_references) == len(self.eval_file_query_names)
        
    def read_dialogue_summarization(self):
        print(("Reading dialogue as turns from {}/{}".format(self.root, self.mode)))
        features = []

        for file_name in tqdm(self.file_names):
            file_path = f"{self.root}/{self.mode}/{file_name}.json"
            print(file_path)

            sent_mapping_dict: Dict[str, int] = {}
            dialogue: List[str] = []

            with open(file_path) as f:
                session = json.load(f)

            # Make sent_mapping idx, dialogue append
            for turn in session['meeting_transcripts']:
                for sentence in turn['sentence_level_content']:
                    dialogue.append(clean_data(turn['speaker'].lower() + ': ' + tokenize(sentence['dialogue_sentence'])))
                    sent_mapping_dict[f"{sentence['turn_index']}-{sentence['sent_index']}"] = len(sent_mapping_dict)
            
            oracle_data_list = list()
            
            for oracle_path in config.oracle_path_list:
                with open(f"{oracle_path}/{file_name}.json", 'r') as f:
                    oracle_data = json.load(f)
                    oracle_data_list.append(oracle_data)

            queries: List[str] = []
            summaries: List[str] = []
            
            oracle_cls_ids_list: list = []
            
            # FIXME: have to develop
            def get_oracle_cls_ids(oracle, query, idx):
                oracle_info = oracle[f'{query}_query_list'][idx]['oracle_info']
                oracle_cls_ids = list(map(
                    # get reverse idx(order of sent)
                    lambda x: sent_mapping_dict[f"{x['turn_index']}-{x['sent_index']}"],
                    oracle_info
                ))
                return oracle_cls_ids
            
            # get text from querys
            for query in ['general', 'specific']:
                for query_idx ,query_info in enumerate(session['explainable_qmsum'][f'{query}_query_list']):
                    queries.append(clean_data(tokenize(query_info['query'])))
                    summaries.append(tokenize(query_info['answer']))
                    
                    tmp_oracle_cls_ids = [get_oracle_cls_ids(oracle, query, query_idx) for oracle in oracle_data_list]
                    oracle_cls_ids_list.append(tmp_oracle_cls_ids)

            assert len(queries) == len(summaries) == len(oracle_cls_ids_list)

            for query, summary, oracle_cls_ids in zip(queries, summaries, oracle_cls_ids_list):
                retriever_inputs: dict = self.tokenize_retriever(dialogue, query, oracle_cls_ids)
                generator_inputs: dict = self.tokenize_generator(dialogue, query, summary)
                features.append((retriever_inputs, generator_inputs))

        return features
        
    def load_features_from_cache(self):
        """Save or Load features from cache file
        """
        print("cached feature file address", self.cached_dataset)
        if os.path.exists(self.cached_dataset) and not config.overwrite_cache:
            print("Loading features from cached file {}".format(self.cached_dataset))
            self.features = torch.load(self.cached_dataset)
        else:
            self.features = self.read_dialogue_summarization()
            print("Saving features into cached file {}".format(self.cached_dataset))
            torch.save(self.features, self.cached_dataset)

    def tokenize_retriever(self, text, query, oracle_cls_ids):
        tokenized_query = self.retriever_tokenizer(query).input_ids
        tokenized_sentence_list = [self.retriever_tokenizer(turn).input_ids for turn in text]

        input_ids_list:List[List[int]] = []
        cls_ids:List[int] = []
        oracle_list: List[List[int]] = []
        idx_offset = 0
        turn_id = 0
        list_id = 0

        # make chunks with sentences
        while turn_id < len(tokenized_sentence_list) and list_id < config.max_chunks:
            # text
            input_ids = []

            # Append query
            input_ids.extend(tokenized_query)

            # Append each dialogue until chunk is (almost)full
            while turn_id < len(tokenized_sentence_list):
                tokenized_sentence = tokenized_sentence_list[turn_id]
                # exceed max length
                if len(input_ids) + len(tokenized_sentence) > config.max_retrieval_len:
                    # stop at first
                    if len(input_ids) == len(tokenized_query):
                        tokenized_sentence = tokenized_sentence[:config.max_retrieval_len - len(input_ids)]
                    else:
                        break
                input_ids.extend(tokenized_sentence)
                
                # Inform end position of each sentence
                cls_ids.append(len(input_ids) - 1 + idx_offset)
                turn_id += 1

            # Append pad token
            num_pad = config.max_retrieval_len - len(input_ids)
            input_ids.extend([self.retriever_tokenizer.pad_token_id] * num_pad)

            # Save
            input_ids_list.append(input_ids)
            idx_offset += config.max_retrieval_len
            list_id += 1

        for oracle_cls in oracle_cls_ids:
            oracle = [
                oracle_id for oracle_id in oracle_cls
                if oracle_id < turn_id and len(text[oracle_id].split(" ")) > 3
            ]
            oracle_list.append(oracle)
        
        # Fill padding
        pad_list(oracle_list)
        retriever_inputs = {
            'input_ids': input_ids_list,
            'cls_ids': cls_ids,
            'oracle': oracle_list,
        }
        
        return retriever_inputs

    def tokenize_generator(self, text, query, summary):
        context_input_ids = []
        labels = None
        context_attention_mask = []

        for turn_id in range(len(text)):
            text_turn = text[turn_id]

            input_dict = self.generator_tokenizer.prepare_seq2seq_batch(
                src_texts=text_turn + " // " + query,
                tgt_texts=summary,
                max_length=config.max_source_len,
                max_target_length=config.max_target_len,
                padding="max_length",
                truncation=True,
            )
            context_attention_mask.append(input_dict.attention_mask)
            context_input_ids.append(input_dict.input_ids)
            if labels is None:
                labels = input_dict.labels
            else:
                assert labels == input_dict.labels

        generator_inputs = {
            'context_input_ids': context_input_ids,
            'context_attention_mask': context_attention_mask,
            'labels': labels
        }
        if labels is None:
            raise ValueError(text)

        return generator_inputs
