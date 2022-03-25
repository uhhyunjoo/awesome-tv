import torch
import numpy as np
import os
import cv2 as cv
from pathlib import Path
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchsummary import summary
import torch.optim as optim

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # type: ignore

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_checkpoint(checkpoint_path, model, optimizer, device):
    checkpoint = torch.load(checkpoint_path, map_location = device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    del checkpoint
    return model, optimizer, start_epoch

def save_checkpoint(epoch, model, optimizer, scheduler, loss, ckpt_file_path):
    if scheduler == None:
        state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss': loss}
    else :
        state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'loss': loss}
    torch.save(state, ckpt_file_path)