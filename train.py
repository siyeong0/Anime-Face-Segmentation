import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import numpy as np
import cv2 as cv
from PIL import Image

from network import UNet
from dataset import UNetDataset
from util import seg2img

####################################################################
# Constants and hyper parameters
DATA_PATH  = './data/faces'
SEG_PATH = '../data/segs'
MODEL_PATH = './model'
MODEL_NAME = 'UNet'

INPUT_LEN = 512

LEARNING_RATE = 0.0001
EPOCH = 50

TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 8
TEST_BATCH_SIZE = 1

R_TRAIN = 0.88; R_VAL = 0.07
# Define transformer
transformer = transforms.Compose([
            transforms.ToTensor(),])
# Load dataset
total_dataset = UNetDataset(img_path=DATA_PATH,seg_path=SEG_PATH, transform=transformer)
# Split train, validation, test
len_total = len(total_dataset)
len_train = int(len_total * R_TRAIN); len_val = int(len_total * R_VAL); len_test = len_total - len_train - len_val
train_dataset, validation_dataset, test_dataset = random_split(total_dataset, [len_train, len_val, len_test])
# Build loaders
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(validation_dataset, batch_size=VAL_BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True)
# Build Model :: In: 3x512x512 -> Out: 7x512x512
model = UNet()
model.load_state_dict(torch.load('save/UNet.pth'))
if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.98)
criterion = nn.BCEWithLogitsLoss()

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        img, seg_target = data
        img = img.cuda()
        seg_target = seg_target.cuda()
        
        optimizer.zero_grad()
        
        pred_seg = model(img)
        loss = criterion(pred_seg, seg_target)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
                
        if batch_idx % 20 == 0:
            scheduler.step()
        
        if batch_idx % 20 == 0:
            print('Train Epoch: {:>6} [{:>6}/{:>6} ({:>2}%)]\tLoss: {:.6f}\t\t lr: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                int(100. * batch_idx / len(train_loader)), loss.item() / len(data), float(scheduler.get_lr()[0])))
    print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss / len(train_loader.dataset)))
    
def validation():
    model.eval()
    val_loss= 0
    with torch.no_grad():
        for data in val_loader:
            img, seg_target = data
            img = img.cuda()
            seg_target = seg_target.cuda()
            pred_seg = model(img)
            
            # sum up batch loss
            val_loss += criterion(pred_seg, seg_target).item()
            
        
    print('====> Test set loss: {:.8f}'.format(val_loss / len(val_loader.dataset)))

for epoch in range(EPOCH):
    train(epoch)
    validation()
    
    torch.save(model.state_dict(), MODEL_PATH+'/'+MODEL_NAME+f'_ep{epoch}'+'.pth')
    

torch.save(model.state_dict(), MODEL_PATH+'/'+MODEL_NAME+'.pth')

with torch.no_grad():
    model.eval()
    for data in test_loader:
        img, seg_target = data
        img = img.cuda()
        seg_target = seg_target.cuda()
        pred_seg = model(img)
        
        pred_seg = pred_seg.cpu().numpy()
        for i in range(TEST_BATCH_SIZE):
            img = seg2img(np.moveaxis(pred_seg[i],0,2))
            cv.imwrite(f'result{i}.png',img)
        break