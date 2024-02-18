#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os,sys
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.cuda as cuda
from torch.utils.data import DataLoader,DistributedSampler
from torch import distributed as dist
from torch import multiprocessing as mp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "../..")  # Assuming "pipeline" is in the parent directory
sys.path.append(parent_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from detector.t5_sentinel.model import Sentinel
from detector.t5_sentinel.utilities import predict,device_initializer,_all_reduce_dict
from detector.t5_sentinel.__init__ import config


import json
import torch.utils as utils
if config.backbone.name=='mt5-large':
    from transformers import MT5Tokenizer as Tokenizer
elif config.backbone.name=='flan-t5-large':
    from transformers import AutoTokenizer as Tokenizer
elif config.backbone.name=='umt5-base' :
    from transformers import AutoTokenizer as Tokenizer
else:
    from transformers import T5TokenizerFast as Tokenizer



current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "../..")  # Assuming "pipeline" is in the parent directory
sys.path.append(parent_dir)
from torch import Tensor
from typing import Tuple

class Dataset(utils.data.Dataset):
    '''
    Dataset for loading text from different large language models.

    Attributes:
        corpus (list[str]): The corpus of the dataset.
        label (list[str]): The labels of the dataset.
        tokenizer (Tokenizer): The tokenizer used.
    '''
    def __init__(self, partition: str, selectedDataset: Tuple[str] = ('Human', 'ChatGPT', 'cohere', 'davinci','bloomz','dolly'),DatasetName = "semeval",taskType = "A",language = "single"):
        super().__init__()
        # if DatasetName != "semeval":
        #     self.corpus, self.label = [], []
        #     filteredDataset = [item for item in config.dataset if item.label in selectedDataset]
        #     for item in filteredDataset:
        #         with open(f'{item.root}/{partition}.jsonl', 'r') as f:
        #             for line in f:
        #                 if item.label == 'LLaMA':
        #                     words = json.loads(line)['text'].split()
        #                     continuation = words[75:]
        #                     if len(continuation) >= 42:
        #                         self.corpus.append(' '.join(continuation[:256]))
        #                         self.label.append(item.token)
        #                 else:
        #                     self.corpus.append(json.loads(line)['text'])
        #                     self.label.append(item.token)
        # else:
            # 处理semeval数据集
        if taskType == "A":
            # A类子任务
            self.corpus, self.ids = [], []
            if language == "single":
                print("task A and single language")
                data_dir = f"/home/yaoxy/T5-Sentinel-public-main/data/split/SemEval2024-Task8-test/subtaskA_monolingual.jsonl"
                with open (data_dir,'r') as f:
                    for line in f:
                        text= 'Discern whether the following text is authored by a human or a machine. If human-written, respond with <extra_id_0>; if machine-generated, respond with <extra_id_1>:\'{0}\''.format(json.loads(line)['text'])
                        self.corpus.append(text)
                        self.ids.append(json.loads(line)['id'])
                # self.corpus = self.corpus[:20]
                # self.ids = self.ids[:20]
                self.tokenizer: Tokenizer = Tokenizer.from_pretrained(f"/home/yaoxy/T5-Sentinel-public-main/detector/t5_sentinel/{config.backbone.name}", model_max_length=config.backbone.modelMaxLen)
            else:
                data_dir = f"/home/yaoxy/T5-Sentinel-public-main/data/split/SemEval2024-Task8-test/subtaskA_multilingual.jsonl"
                # 多语言用英文提示
                with open (data_dir,'r') as f:
                    for line in f:
                        text= 'Discern whether the following text is authored by a human or a machine. If human-written, respond with <extra_id_0>; if machine-generated, respond with <extra_id_1>:\'{0}\''.format(json.loads(line)['text'])
                        self.corpus.append(text)
                        self.ids.append(json.loads(line)['id'])
                # self.corpus = self.corpus[:20]
                # self.ids = self.ids[:20]
                # 指定要添加的额外标记的数量
                extra_ids = 2
                # 指定额外的特殊标记 0是human，1是mechine
                additional_special_tokens = ["<extra_id_0>", "<extra_id_1>"]
    
                self.tokenizer: Tokenizer = Tokenizer.from_pretrained(f"/home/yaoxy/T5-Sentinel-public-main/detector/t5_sentinel/{config.backbone.name}", 
                                                extra_ids=extra_ids,
                                                additional_special_tokens=additional_special_tokens,
                                                model_max_length=config.backbone.modelMaxLen)
    
                # 获取额外标记的 id
                sentinel_token_ids = self.tokenizer.get_sentinel_token_ids()
                print("Sentinel Token IDs:", sentinel_token_ids)

                # 获取额外标记的字符串
                sentinel_tokens = self.tokenizer.get_sentinel_tokens()
                print("Sentinel Tokens:", sentinel_tokens)

        
        
    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.corpus[idx], self.ids[idx]
    
    def collate_fn(self, batch: Tuple[str, str]) -> Tuple[Tensor, Tensor, Tensor]:
        corpus, id = zip(*batch)
        corpus = self.tokenizer.batch_encode_plus(corpus, padding=config.tokenizer.padding, truncation=config.tokenizer.truncation, return_tensors=config.tokenizer.return_tensors)
        # label = self.tokenizer.batch_encode_plus(label, padding=config.tokenizer.padding, truncation=config.tokenizer.truncation, return_tensors=config.tokenizer.return_tensors)
        return corpus.input_ids, corpus.attention_mask, id



def get_dataloader(distributed:bool):
    test_loader = DataLoader(
            test_dataset := Dataset('test',DatasetName=config.dataloader.dataset_name,
                taskType=config.dataloader.taskType,
                language=config.dataloader.language),
            collate_fn=test_dataset.collate_fn,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            shuffle=False
        )
    return test_loader

test_dataset = Dataset('test',DatasetName=config.dataloader.dataset_name,
                taskType=config.dataloader.taskType,
                language=config.dataloader.language)
label_map = {
    "<extra_id_0>":0,
    "<extra_id_1>":1
}
distributed=False
test_loader = get_dataloader(distributed)
device = device_initializer()
model = Sentinel().cuda().to(device)

cache = f'storage/{config.id}'
if os.path.exists(f'{cache}/state.pt'):
    print(f"load model {cache}")
    state = torch.load(f'{cache}/state.pt')
    model.load_state_dict(state['model'])
selectedDataset = ('Human', 'ChatGPT')
all_ids,all_predictions,all_probabilities = predict(model, test_loader,selectedDataset)
print("start write to file")
with open(f"submit_{config.backbone.name}{config.dataloader.dataset_name}{config.dataloader.taskType}.json","w") as f:
    for id,prediction,probability in zip(all_ids,all_predictions,all_probabilities):
        # print(test_dataset.tokenizer.convert_ids_to_tokens(prediction))
        pred_label = label_map[test_dataset.tokenizer.convert_ids_to_tokens(prediction)]
        line_res = json.dumps({"id":id,"label":pred_label,"probability":probability})
        f.write(line_res + '\n')

print("save success!")

    