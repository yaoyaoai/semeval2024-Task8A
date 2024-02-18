import os
import wandb
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
from torch.utils.data import DataLoader,DistributedSampler
import os,sys
from torch import distributed as dist
from torch import multiprocessing as mp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "../..")  # Assuming "pipeline" is in the parent directory
sys.path.append(parent_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from detector.t5_sentinel.dataset import Dataset
from detector.t5_sentinel.model import Sentinel
from detector.t5_sentinel.utilities import train, validate,device_initializer,_all_reduce_dict
from detector.t5_sentinel.__init__ import config
from detector.t5_sentinel.optim import Adafactor


##############################################################################
# Dataset and Dataloader
##############################################################################
def get_dataset(distributed:bool):
    if distributed:
        # 分布式加载
        # 训练集的分布式采样器
        train_dataset = Dataset(
                partition='train',# 训练或者测试
                DatasetName=config.dataloader.dataset_name, # 不用改
                taskType=config.dataloader.taskType,
                language=config.dataloader.language # 单语言或者多语言
            )
        train_sampler = DistributedSampler(
            dataset := train_dataset
        )
        # DataLoader 使用 DistributedSampler，并指定 batch_size
        train_loader = DataLoader(
            train_dataset,
            collate_fn=train_dataset.collate_fn,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            shuffle=False,  # 在 DistributedSampler 中已经进行了 shuffle
            sampler=train_sampler
        )
        # 验证集不使用
        valid_loader = DataLoader(
        valid_dataset := Dataset('dev',DatasetName=config.dataloader.dataset_name,
                taskType=config.dataloader.taskType,
                language=config.dataloader.language),
        collate_fn=valid_dataset.collate_fn,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        shuffle=False
        )
        
        return train_loader,valid_loader
    else:
        train_loader = DataLoader(
            train_dataset := Dataset('train',DatasetName=config.dataloader.dataset_name,taskType=config.dataloader.taskType,language=config.dataloader.language),
            collate_fn=train_dataset.collate_fn, 
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            shuffle=True,
        )   


        valid_loader = DataLoader(
            valid_dataset := Dataset('dev',DatasetName=config.dataloader.dataset_name,
                taskType=config.dataloader.taskType,
                language=config.dataloader.language),
            collate_fn=valid_dataset.collate_fn,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            shuffle=False
        )
        return train_loader,valid_loader


##############################################################################
# Model, Optimizer, and Scheduler
##############################################################################
def main(rank=None,distributed=False):
    
    
    if distributed:
        # 多卡训练
        # model = nn.DataParallel(model) 效率太低
        world_size = cuda.device_count()
        os.environ["MASTER_ADDR"] = "localhost" # 设置地址和端口
        os.environ["MASTER_PORT"] = "12349"
        # 初始化训练环境
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", rank=rank,
                                    world_size=world_size)
        dist.barrier()
        if dist.get_rank() == 0: # 默认主GPU为0号
            save_model = True
        device = torch.device("cuda", rank)
        train_loader,valid_loader = get_dataset(distributed)
        print(f"Successfully Use distributed training {rank}")
    else:
        save_model = True # 单卡训练 保存模型
        train_loader,valid_loader = get_dataset(distributed)
        device = device_initializer()
        print(f"Use single gpu for training {device}")
    model = Sentinel().cuda().to(device)
    if distributed:
        # 将模型加载至分布式
        model = nn.parallel.DistributedDataParallel(model,device_ids=[device])
    if config.backbone.name.endswith("large"):
       print("using adafactor for optim")
       optimizer = Adafactor(
        model.module.parameters() if distributed else model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False)
 
    else:
        optimizer = optim.AdamW(
        model.module.parameters() if distributed else model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay
        )

    ##############################################################################
    # Task and Cache
    ##############################################################################
    cache = f'storage/{config.id}'
    
    if rank == 0:
        # # 仅主进程进行wandb初始化
        task = wandb.init(
            name=config.id, 
            project='T5llmdetection-yaoxy', 
            resume='allow',
        )
        

        wandb.save('detector/models/arbitrary/__init__.py')
        wandb.save('detector/models/arbitrary/__main__.py')
        wandb.save('detector/models/arbitrary/dataset.py')
        wandb.save('detector/models/arbitrary/model.py')
        wandb.save('detector/models/arbitrary/settings.yaml')
        wandb.save('detector/models/arbitrary/t5types.py')
        wandb.save('detector/models/arbitrary/utilities.py')
        os.path.exists(cache) or os.makedirs(cache)
    if not distributed:
        task = wandb.init(
            name=config.id, 
            project='T5llmdetection-yaoxy', 
            resume='allow',
        )
        

        wandb.save('detector/models/arbitrary/__init__.py')
        wandb.save('detector/models/arbitrary/__main__.py')
        wandb.save('detector/models/arbitrary/dataset.py')
        wandb.save('detector/models/arbitrary/model.py')
        wandb.save('detector/models/arbitrary/settings.yaml')
        wandb.save('detector/models/arbitrary/t5types.py')
        wandb.save('detector/models/arbitrary/utilities.py')
        os.path.exists(cache) or os.makedirs(cache)

    if distributed:
        dist.barrier() # 在这里阻塞一下

    if os.path.exists(f'{cache}/state.pt'):
        print(f"load model {cache}")
        state = torch.load(f'{cache}/state.pt')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        startEpoch = state['epochIter'] + 1
        bestValidationAccuracy = state['validAccuracy']

    else:
        startEpoch = 0
        bestValidationAccuracy = float('-inf')


    selectedDataset = ('Human', 'ChatGPT', 'cohere', 'davinci', 'bloomz', 'dolly') if config.dataloader.taskType == "B" else ('Human', 'ChatGPT')
    ##############################################################################
    # Training and Validation
    ##############################################################################
    for epoch in range(startEpoch, config.epochs):
        if rank == 0:
            tqdm.write('Epoch {}'.format(epoch + 1))
        if not distributed:
            tqdm.write('Epoch {}'.format(epoch + 1))
        learnRate = optimizer.param_groups[0]['lr']
        if distributed:
            train_alone_metrics = train(model, optimizer, train_loader,selectedDataset)
            train_all_metrics = _all_reduce_dict(train_alone_metrics,device)
        elif not distributed:
            train_all_metrics = train(model, optimizer, train_loader,selectedDataset)
        if rank == 0 and distributed:
            valid_metrics = validate(model, valid_loader,selectedDataset) # 主进程才进行验证
        elif not distributed:
            valid_metrics = validate(model, valid_loader,selectedDataset)
        
        train_metrics = {
            "TrainingLoss":train_all_metrics['TrainingLoss'],
            "TrainingAccuracy":train_all_metrics['TrainingCorrect']/train_all_metrics['TrainingBatch']
        }
        if rank == 0 and distributed:
            tqdm.write(f"device {device} update wandb data...")
            wandb.log({
                **train_metrics,
                **valid_metrics,
                'Learning Rate': learnRate
            })
            tqdm.write(f"device {device} complete...")
            tqdm.write('TrainingAccuracy {:.2%}'.format(train_metrics['TrainingAccuracy']))
            tqdm.write('TrainingLoss {:.4f}'.format(train_metrics['TrainingLoss']))
            tqdm.write('ValidationAccuracy {:.2%}'.format(valid_metrics['ValidationAccuracy']))
            # tqdm.write('TrainingMicrof1 {:.4f}'.format(train_metrics.TrainingMicrof1))
            tqdm.write('ValidationMicrof1 {:.2%}'.format(valid_metrics['Validationmicrof1']))
            # tqdm.write('TrainingMacrof1 {:.4f}'.format(train_metrics.TrainingMacrof1))
            tqdm.write('ValidationMacrof1 {:.2%}'.format(valid_metrics['Validationmacrof1']))
            tqdm.write('Learning Rate {:.4f}'.format(learnRate)) 
  
        elif not distributed:
            tqdm.write(f"device {device} update wandb data...")
            wandb.log({
                **train_metrics,
                **valid_metrics,
                'Learning Rate': learnRate
            })
            tqdm.write(f"device {device} complete...")
            tqdm.write('TrainingAccuracy {:.2%}'.format(train_metrics['TrainingAccuracy']))
            tqdm.write('TrainingLoss {:.4f}'.format(train_metrics['TrainingLoss']))
            tqdm.write('ValidationAccuracy {:.2%}'.format(valid_metrics['ValidationAccuracy']))
            # tqdm.write('TrainingMicrof1 {:.4f}'.format(train_metrics.TrainingMicrof1))
            tqdm.write('ValidationMicrof1 {:.2%}'.format(valid_metrics['Validationmicrof1']))
            # tqdm.write('TrainingMacrof1 {:.4f}'.format(train_metrics.TrainingMacrof1))
            tqdm.write('ValidationMacrof1 {:.2%}'.format(valid_metrics['Validationmacrof1']))
            tqdm.write('Learning Rate {:.4f}'.format(learnRate)) 
        if distributed:
            if rank == 0:
                checkpoint_dir = f"{cache}/{epoch}"
                os.makedirs(checkpoint_dir)
                checkpoint = {
                    "epochIter": epoch,
                    "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "validAccuracy": valid_metrics['ValidationAccuracy'],
                }
                tqdm.write(f"device {device} saving check point model")
                # bestValidationAccuracy = valid_metrics['ValidationAccuracy']
                torch.save(checkpoint, f'{checkpoint_dir}/state.pt')
                tqdm.write('Checkpoint Saved!')
            dist.barrier()
        else:
            checkpoint_dir = f"{cache}/{epoch}"
            os.makedirs(checkpoint_dir)
            checkpoint = {
                "epochIter": epoch,
                "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "validAccuracy": valid_metrics['ValidationAccuracy'],
            }
            tqdm.write(f"device {device} saving check point model")
            # bestValidationAccuracy = valid_metrics['ValidationAccuracy']
            torch.save(checkpoint, f'{checkpoint_dir}/state.pt')
            tqdm.write('Checkpoint Saved!')

if __name__ == "__main__":
    if cuda.device_count() > 1:
        gpus = torch.cuda.device_count()
        mp.spawn(main, args=(True,), nprocs=gpus)
    else:
        print("No distributed training start!")
        main()