#!/usr/bin/env python
# -*- coding:utf-8 -*-

import json
import torch.utils as utils
from detector.t5_sentinel.__init__ import config
if config.backbone.name=='mt5-large' or config.backbone.name=='mt5-small' or config.backbone.name=='mt5-xl':
    from transformers import MT5Tokenizer as Tokenizer
elif config.backbone.name=='flan-t5-large':
    from transformers import AutoTokenizer as Tokenizer
elif config.backbone.name=='umt5-base' :
    from transformers import AutoTokenizer as Tokenizer
else:
    from transformers import T5TokenizerFast as Tokenizer

import os,sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "../..")  # Assuming "pipeline" is in the parent directory
sys.path.append(parent_dir)
from detector.t5_sentinel.__init__ import config
from torch import Tensor
from typing import Tuple
from torch.utils.data import DataLoader


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
        if DatasetName != "semeval":
            self.corpus, self.label = [], []
            filteredDataset = [item for item in config.dataset if item.label in selectedDataset]
            for item in filteredDataset:
                with open(f'{item.root}/{partition}.jsonl', 'r') as f:
                    for line in f:
                        if item.label == 'LLaMA':
                            words = json.loads(line)['text'].split()
                            continuation = words[75:]
                            if len(continuation) >= 42:
                                self.corpus.append(' '.join(continuation[:256]))
                                self.label.append(item.token)
                        else:
                            self.corpus.append(json.loads(line)['text'])
                            self.label.append(item.token)
        else:
            # 处理semeval数据集
            if taskType == "A":
                # A类子任务
                self.corpus, self.label = [], []
                if language == "single":
                    data_dir = f"/home/yaoxy/T5-Sentinel-public-main/data/split/Semeval/SubtaskA/subtaskA_{partition}_monolingual.jsonl"
                    with open (data_dir,'r') as f:
                        for line in f:
                            text= 'Discern whether the following text is authored by a human or a machine. If human-written, respond with <extra_id_0>; if machine-generated, respond with <extra_id_1>:\'{0}\''.format(json.loads(line)['text'])
                            # text= '\'{0}\'Discern whether the above text is authored by a human or a machine. If human-written, respond with <extra_id_0>; if machine-generated, respond with <extra_id_1>'.format(json.loads(line)['text'])
                            # print(text)
                            self.corpus.append(text)
                            self.label.append(f"<extra_id_{json.loads(line)['label']}>")
                else:
                    # data_dir = f"/home/yaoxy/T5-Sentinel-public-main/data/split/Semeval/SubtaskA/subtaskA_{partition}_multilingual.jsonl"
                    data_dir = f"/home/yaoxy/T5-Sentinel-public-main/data/split/Semeval/SubtaskA/subtaskA_{partition}_monolingual.jsonl"
                    # 多语言用英文提示
                    with open (data_dir,'r') as f:
                        for line in f:
                            text= 'Discern whether the following text is authored by a human or a machine. If human-written, respond with <extra_id_0>; if machine-generated, respond with <extra_id_1>:\'{0}\''.format(json.loads(line)['text'])
                            self.corpus.append(text)
                            self.label.append(f"<extra_id_{json.loads(line)['label']}>")
            if taskType == "B":
                # B类子任务
                self.corpus, self.label = [], []
                data_dir = f"/home/yaoxy/T5-Sentinel-public-main/data/split/Semeval/SubtaskB/subtaskB_{partition}.jsonl"
                with open (data_dir,'r') as f:
                    for line in f:
                        self.corpus.append(json.loads(line)['text'])
                        self.label.append(f"<extra_id_{json.loads(line)['label']}>")
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
        return self.corpus[idx], self.label[idx]
    
    def collate_fn(self, batch: Tuple[str, str]) -> Tuple[Tensor, Tensor, Tensor]:
        corpus, label = zip(*batch)
        corpus = self.tokenizer.batch_encode_plus(corpus, padding=config.tokenizer.padding, truncation=config.tokenizer.truncation, return_tensors=config.tokenizer.return_tensors)
        label = self.tokenizer.batch_encode_plus(label, padding=config.tokenizer.padding, truncation=config.tokenizer.truncation, return_tensors=config.tokenizer.return_tensors)
        return corpus.input_ids, corpus.attention_mask, label.input_ids

if __name__ == "__main__":
    train_loader = DataLoader(
    train_dataset := Dataset('train'),
    collate_fn=train_dataset.collate_fn, 
    batch_size=1,
    num_workers=1,
    shuffle=True
    )
    
    for batch in train_loader:
    
        print(batch)