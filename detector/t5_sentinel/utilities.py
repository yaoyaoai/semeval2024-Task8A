import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from detector.t5_sentinel.__init__ import config
from detector.t5_sentinel.t5types import SentinelOutput
from typing import Tuple
from sklearn.metrics import precision_recall_fscore_support
import torch


def train(
    model: nn.Module,
    optimizer: nn.Module,
    dataloader: DataLoader,
    selectedDataset: Tuple[str] = (
        "Human",
        "ChatGPT",
        "cohere",
        "davinci",
        "bloomz",
        "dolly",
    ),
) -> Tuple[float, float]:
    model.train()
    accumulatedLoss, accumulatedCorrect, accumulatedBatchSize = 0, 0, 0
    progress = tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Training", ncols=120
    )

    filteredDataset = [item for item in config.dataset if item.label in selectedDataset]
    all_labels = []
    all_predictions = []
    for i, (corpus_ids, corpus_mask, label_ids) in progress:
        output: SentinelOutput = model.forward(
            corpus_ids.cuda(), corpus_mask.cuda(), label_ids.cuda()
        )
        loss, probabilities, predictions = (
            output.huggingface.loss,
            output.probabilities,
            [],
        )
        for argmaxIndex in probabilities.argmax(dim=-1):
            predictions.append(filteredDataset[argmaxIndex].token_id)

        accumulatedLoss += loss.mean().item()
        accumulatedCorrect += sum(
            [
                1 if prediction == label_id[0] else 0
                for prediction, label_id in zip(predictions, label_ids.tolist())
            ]
        )
        accumulatedBatchSize += config.dataloader.batch_size
        
        all_labels.extend([label[0] for label in label_ids.tolist()])
        # print([label[0] for label in label_ids.tolist()])
        all_predictions.extend(predictions)
        # print(predictions,[label[0] for label in label_ids.tolist()])
        loss.mean().backward()
        if (
            accumulatedBatchSize >= config.optimizer.batch_size
            or i == len(dataloader) - 1
        ):
            optimizer.step()
            optimizer.zero_grad()
            accumulatedBatchSize = 0

        progress.set_postfix(
            {
                "loss": "{:04f}".format(accumulatedLoss / (i + 1)),
                "accuracy": "{:04%}".format(
                    accumulatedCorrect / ((i + 1) * config.dataloader.batch_size)
                ),
            }
        )

    progress.close()
    # 计算 micro 精确度、召回率、F1 分数
    # micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
    #     all_labels, all_predictions, average="micro"
    # )

    # 计算 macro 精确度、召回率、F1 分数
    # macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    #     all_labels, all_predictions, average="macro"
    # )
    trainLoss = accumulatedLoss / len(dataloader)
    # trainAccuracy = accumulatedCorrect / (
    #     len(dataloader) * config.dataloader.batch_size
    # )
    return {
        "TrainingLoss": trainLoss,
        "TrainingCorrect": accumulatedCorrect,
        "TrainingBatch":len(dataloader) * config.dataloader.batch_size
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    selectedDataset: Tuple[str] = (
        "Human",
        "ChatGPT",
        "cohere",
        "davinci",
        "bloomz",
        "dolly",
    ),
) -> dict:
    model.eval()
    accumulatedCorrect = 0
    # 单卡总批数
    progress = tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Validating", ncols=120
    )

    filteredDataset = [item for item in config.dataset if item.label in selectedDataset]
    # print(filteredDataset)
    all_labels = []
    all_predictions = []
    for i, (corpus_ids, corpus_mask, label_ids) in progress:
        output: SentinelOutput = model.forward(
            corpus_ids.cuda(), corpus_mask.cuda(), label_ids.cuda()
        )
        probabilities, predictions = output.probabilities, []
        for argmaxIndex in probabilities.argmax(dim=-1):
            predictions.append(filteredDataset[argmaxIndex].token_id)
        all_labels.extend([label[0] for label in label_ids.tolist()])
        # print([label[0] for label in label_ids.tolist()])
        all_predictions.extend(predictions)

        accumulatedCorrect += sum(
            [
                1 if prediction == label_id[0] else 0
                for prediction, label_id in zip(predictions, label_ids.tolist())
            ]
        )
        # 目前这一批达到的准确率
        progress.set_postfix(
            {
                "accuracy": "{:04%}".format(
                    accumulatedCorrect / ((i + 1) * config.dataloader.batch_size)
                )
            }
        )
    # 单卡计算
    progress.close()
    # 计算 micro 精确度、召回率、F1 分数
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average="micro"
    )

    # 计算 macro 精确度、召回率、F1 分数
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average="macro"
    )

    validAccuracy = accumulatedCorrect / (
        len(dataloader) * config.dataloader.batch_size
    )
    return {
        "Validationmicrof1": micro_f1,
        "Validationmacrof1": macro_f1,
        "Validationmicroprecision": micro_precision,
        "Validationmicrorecall": micro_recall,
        "Validationmacroprecision": macro_precision,
        "Validationmacrorecall": macro_recall,
        "ValidationAccuracy": validAccuracy,
    }

def predict(
    model: nn.Module,
    dataloader: DataLoader,
    selectedDataset: Tuple[str] = (
        "Human",
        "ChatGPT",
        "cohere",
        "davinci",
        "bloomz",
        "dolly",
    ),
) -> dict:
    model.eval()
    # accumulatedCorrect = 0
    # 单卡总批数
    progress = tqdm(
        enumerate(dataloader), total=len(dataloader), desc="predicting", ncols=120
    )

    filteredDataset = [item for item in config.dataset if item.label in selectedDataset]
    # print(filteredDataset)
    all_ids = []
    all_predictions = []
    all_probabilities = []
    for i, (corpus_ids, corpus_mask, ids) in progress:
        output: SentinelOutput = model.forward(
            corpus_ids.cuda(), corpus_mask.cuda()
        )
        probabilities, predictions= output.probabilities, []
        for argmaxIndex in probabilities.argmax(dim=-1):
            predictions.append(filteredDataset[argmaxIndex].token_id)
        for probability in probabilities:
            index_list = [filteredDataset[index].label for index, value in enumerate(probability)]
            value_list = [value.item() for index, value in enumerate(probability)]
            probability_dict = dict(zip(index_list,value_list))
            all_probabilities.append(str(probability_dict))
        all_ids.extend([id for id in ids])
        # print([label[0] for label in label_ids.tolist()])
        
        all_predictions.extend(predictions)

        # accumulatedCorrect += sum(
        #     [
        #         1 if prediction == label_id[0] else 0
        #         for prediction, label_id in zip(predictions, label_ids.tolist())
        #     ]
        # )
        # # 目前这一批达到的准确率
        # progress.set_postfix(
        #     {
        #         "accuracy": "{:04%}".format(
        #             accumulatedCorrect / ((i + 1) * config.dataloader.batch_size)
        #         )
        #     }
        # )
    # 单卡计算
    progress.close()
    print(all_ids,all_predictions,all_probabilities)
    return all_ids,all_predictions,all_probabilities

def device_initializer():
    """
    This function initializes the running device information when the program runs for the first time
    :return: cpu or cuda
    """
    # logger.info(msg="Init program, it is checking the basic setting.")
    device_dict = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device(device="cuda")
        is_init = torch.cuda.is_initialized()
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(device=device)
        device_cap = torch.cuda.get_device_capability(device=device)
        device_prop = torch.cuda.get_device_properties(device=device)
        device_dict["is_init"] = is_init
        device_dict["device_count"] = device_count
        device_dict["device_name"] = device_name
        device_dict["device_cap"] = device_cap
        device_dict["device_prop"] = device_prop
        # logger.info(msg=device_dict)
    else:
        # logger.warning(msg="The device is using cpu.")
        device = torch.device(device="cpu")
    return device

def _all_reduce_dict(d, device):
    # wrap in tensor and use reduce to gpu0 tensor
    output_d = {}
    for (key, value) in sorted(d.items()):
        tensor_input = torch.tensor([[value]]).to(device)
        torch.distributed.all_reduce(tensor_input)
        output_d[key] = tensor_input.item()   
    return output_d