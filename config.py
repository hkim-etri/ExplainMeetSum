import os
import torch

CACHE_DATASET_NAME = 'my_dataset'
OUTPUT_DIR_NAME = 'my_model'


class Config():
    def __init__(self):
        self.retriever_name_or_path = 'roberta-base'
        self.generator_name_or_path = 'facebook/bart-large'
        
        # Number of retriever(roberta-base)
        self.num_retriever = 2
        # oracle path
        self.oracle_path_list = [
            'data/oracle/rog_oracle',
            'data/oracle/ces_oracle'
        ]
        self.extractor_top_k = [15, 15]

        # Training configuration.
        self.num_epochs = 15
        self.max_grad_norm = 1.0
        self.cls_lr = 5e-5
        self.weight_decay = 0.0
        self.train_batch_size = 8
        self.gradient_accumulation_steps = 8

        assert self.train_batch_size % self.gradient_accumulation_steps == 0

        # Evaluation configuration
        self.eval_batch_size = 1
        self.test_batch_size = 1

        # Miscellaneous
        self.num_workers = 8
        self.seed = 0
        self.gpu = torch.cuda.is_available()

        # Retriever
        self.max_retrieval_len = 512
        self.max_source_len = 300
        self.max_chunks = 300
        self.hybrid_train = True
        self.consistency_alpha = 1
        self.loss_alpha = 1

        # Generator
        self.gen_lr = 5e-6
        self.beam_size = 5
        self.min_length = 100
        self.max_target_len = 600
        self.no_repeat_ngram_size = 2
        self.length_penalty = 1
        # For RAG
        self.detach_generator_consistency = True
        
        # Dataset
        # root directory of dataset
        self.dataset = 'data/ExplainMeetSum'
        # cache file of dataset
        self.cached_dataset = CACHE_DATASET_NAME
        self.overwrite_cache = False

        # Model
        # model save
        output_dir_name = OUTPUT_DIR_NAME
        # save only top_k model
        self.save_top_k = 5
        self.save_model_dir = self.model_specific_dir(f'MultiDyle/outputs/{output_dir_name}/saved_model')
        self.tmp_dir = self.model_specific_dir(f'MultiDyle/outputs/{output_dir_name}/temp_results')

        # dyle ckpt directory
        self.dyle_ckpt_dir = 'MultiDyle/dyle/'
        # trained model directory
        self.eval_model_dir = 'MultiDyle/outputs/multidyle-best-model/'
        
        # num of retriever should be same as num of oracle and length of top_k list
        assert self.num_retriever == len(self.oracle_path_list) == len(self.extractor_top_k)

    def model_specific_dir(self, root):
        """ model-normalization """
        os.makedirs(root, exist_ok=True)
        return root
