import os

from torch import Tensor
from transformers.modeling_outputs import Seq2SeqLMOutput
from pydantic import BaseModel, validator
import typing

class BackboneField(BaseModel):
    name: str
    modelMaxLen: int

    @validator('name')
    def verify_name(cls, v: str):
        if v not in ['models-small', 'mt5-large', 't5-large', 'flan-t5-large','umt5-base','t5-small','mt5-xl']:
            raise ValueError(f'Backbone using {v} is not supported!')
        return v


class DatasetItem(BaseModel):
    label: str
    token: str
    token_id: int
    root: str

    @validator('root')
    def verify_root(cls, v: str):
        # if not os.path.exists(v):
        #     raise ValueError(f'Directory {v} does not exist!')
        return v


class DataloaderField(BaseModel):
    batch_size: int
    num_workers: int
    dataset_name: str
    taskType: str
    language: str
    @validator('num_workers')
    def verify_num_workers(cls, v: int):
        if v < 1 or v > os.cpu_count():
            raise ValueError(f'Number of workers {v} is not supported!')
        return v


class TokenizerField(BaseModel):
    padding: bool
    truncation: bool
    return_tensors: str

    @validator('return_tensors')
    def verify_return_tensors(cls, v: str):
        if v not in ['pt', 'tf', 'np']:
            raise ValueError(f'Returning tensors with {v} is not supported!')
        return v


class OptimizerField(BaseModel):
    lr: float
    weight_decay: float
    batch_size: int


class Config(BaseModel):
    '''
    @note: 
        - If mode is set to 'interpret', all hidden states and attention weights will be returned.
    '''
    id: str
    mode: str
    epochs: int
    backbone: BackboneField
    dataset: typing.List[DatasetItem]
    dataloader: DataloaderField
    tokenizer: TokenizerField
    optimizer: OptimizerField

    @validator('mode')
    def verify_mode(cls, v: str):
        if v not in ['training', 'interpret']:
            raise ValueError(f'Mode {v} is not supported!')
        return v

    @validator('dataset')
    def verify_dataset(cls, v: typing.List[DatasetItem]):
        labels, tokens, roots = set(), set(), set()
        for item in v:
            if item.label in labels:
                raise ValueError(f'Label {item.label} is not unique!')
            labels.add(item.label)
            if item.token in tokens:
                raise ValueError(f'Token {item.token} is not unique!')
            tokens.add(item.token)
            if item.root in roots:
                raise ValueError(f'Root {item.root} is not unique!')
            roots.add(item.root)
        return v


class SentinelOutput(BaseModel):
    huggingface: Seq2SeqLMOutput
    probabilities: Tensor

    class Config:
        arbitrary_types_allowed = True
