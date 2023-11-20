import os
from contextlib import nullcontext
from  datetime import timedelta
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader
import scanpy as sc
from utils import *
from dataset_wrappers import PretrainingDataset
from performer_pytorch import PerformerLM
import logging

# 提取环境变量
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

is_master = RANK == 0
device = torch.device("cuda", LOCAL_RANK)
logging.basicConfig(level = logging.INFO if RANK in [-1, 0] else logging.WARN)


SEED = 0
TIMEOUT = 1200
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
EPOCHS = 8
GRADIENT_ACCUMULATION = 60
VALIDATE_EVERY = 1
NUM_BINS = 7

@record
def main():
    seed_all(SEED + RANK)

    # 设置当前device
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(backend = 'nccl', timeout = timedelta(seconds = TIMEOUT))


    # region: data
    train_data = sc.read_h5ad("../train_data.h5ad")
    test_data = sc.read_h5ad("../test_data.h5ad")
    train_dataset = PretrainingDataset(train_data, num_bins = NUM_BINS)
    test_dataset = PretrainingDataset(test_data, num_bins = NUM_BINS)

    train_sampler = DistributedSampler(train_dataset)
    test_sampler = SequentialDistributedSampler(test_dataset, batch_size = BATCH_SIZE)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, sampler = train_sampler)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, sampler = test_sampler)
    # endregion

    # region
    model = PerformerLM(num_bins = NUM_BINS, max_seq_len = 1e6, dim = 200, heads = 10, depth = 6)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    # endregion

    model = DDP(model, device_ids = [LOCAL_RANK], output_device = LOCAL_RANK)

    optimizer = Adam(model.parameters(), lr = LEARNING_RATE)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps = 15,
        cycle_mult = 2,
        max_lr = LEARNING_RATE,
        min_lr = 1e-6,
        warmup_steps = 5,
        gamma = 0.9
    )

    dist.barrier()

    for epoch in range(EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        dist.barrier()

        train_loss = 0.0
        train_acc = 0.0
        
        # 梯度累加矫正
        adjusted_factor = GRADIENT_ACCUMULATION
        for i, (data, labels) in enumerate(train_loader):
            i += 1
            data, labels = data.to(device), labels.to(device)
            # 这里跟常规的梯度累加不同，如果dataloader的长度不能被K整除，那么最后更新一次参数的时候梯度会偏小，即除以的K偏大
            # 因此这里做了一个矫正
            if i == (len(train_loader) // GRADIENT_ACCUMULATION) * GRADIENT_ACCUMULATION + 1:
                adjusted_factor = len(train_loader) % GRADIENT_ACCUMULATION

            my_context = model.no_sync if i % GRADIENT_ACCUMULATION != 0 else nullcontext
            with my_context():
                logits = model(data)
                # logits: [B, N, NUM_BINS+1], labels: [B, N]
                loss = loss_fn(logits.reshape(-1, NUM_BINS+1), labels.reshape(-1))
                adjusted_loss = loss / adjusted_factor
                adjusted_loss.backward()
            if i % GRADIENT_ACCUMULATION == 0 or (i == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                optimizer.step()
                optimizer.zero_grad()

            if i % 5000 == 0:
                logging.info(f"current loss: {loss:.4f}")
                
            train_loss += loss.item()
            train_acc = mlm_accuracy(logits, labels)

        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_acc / len(train_loader)
        train_loss = get_reduced(train_loss, LOCAL_RANK, 0, WORLD_SIZE)
        train_acc = get_reduced(train_acc, LOCAL_RANK, 0, WORLD_SIZE)

        logging.info(f'    ==  Epoch: {epoch} | Training Loss: {train_loss:.4f} | Accuracy: {train_acc:6.4f}%  ==')

        dist.barrier()
        scheduler.step()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            dist.barrier()

            test_loss = 0.0
            with torch.no_grad():
                for i, (data, labels) in enumerate(test_loader):
                    i += 1
                    data, labels = data.to(device), labels.to(device)
                    logits = model(data)
                    loss = loss_fn(logits.reshape(-1, NUM_BINS+1), labels.reshape(-1))

                    test_loss += loss.item()
                    test_acc = mlm_accuracy(logits, labels)
                    if i % 5000 == 0:
                        logging.info(f"current loss: {loss:.4f}")

                test_loss = test_loss / len(test_loader)
                test_acc = 100 * test_acc / len(test_loader)
                test_loss = get_reduced(test_loss, LOCAL_RANK, 0, WORLD_SIZE)
                test_acc = get_reduced(test_acc, LOCAL_RANK, 0, WORLD_SIZE)

                logging.info(f'    ==  Epoch: {i} | Validation Loss: {test_loss:.4f} | Accuracy: {test_acc:6.4f}%  ==')
                save_best_ckpt(i, model, optimizer, scheduler, test_loss, "pretrain", "ckpts/")

if __name__ == "__main__":
    main()