#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

""" Constant """
worker = 4 # processes of dataloader
num_classes = 10 # cifar-10
width = 32 # image width
height = 32 # image height
channel = 3 # image channel

""" DataLoader """
class CustomDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transoforms=None):
        buff = []
        bins = sorted(glob.glob(os.path.join(img_dir, '**/*.bin'), recursive=True))
        for bin in bins:
            with open(bin, mode='rb') as f:
                buff.extend(f.read())
        self.img_stream = np.array(buff).astype(np.uint8)
        self.num_length = 1 + width * height * channel
        self.num_stream = len(self.img_stream) // self.num_length

    def __len__(self):
        return self.num_stream

    def __getitem__(self, idx):
        offset = self.num_length * idx
        ann = self.img_stream[offset:offset+1]
        img = self.img_stream[offset+1:offset+self.num_length]
        img = img.reshape(channel, height, width)
        img = img[::-1, :, :].astype(np.float32)
        return torch.from_numpy(img), int(ann)

""" Argument Parser """
def parseArgument():
    arg = sys.argv
    num = len(arg)
    ret = { }
    nop = False
    for i in range(1, num):
        if nop:
            nop = False
            continue
        # parameter: images
        if arg[i] == '--images' and i + 1 < num:
            nop = True
            ret['images'] = arg[1+i]
        elif arg[i].startswith('--images='):
            ret['images'] = arg[i][len('--images='):]
        # parameter: anns
        elif arg[i] == '--anns' and i + 1 < num:
            nop = True
            ret['anns'] = arg[1+i]
        elif arg[i].startswith('--anns='):
            ret['anns'] = arg[i][len('--anns='):]
        # parameter: train_log_dir
        elif arg[i] == '--train_log_dir' and i + 1 < num:
            nop = True
            ret['train_log_dir'] = arg[1+i]
        elif arg[i].startswith('--train_log_dir='):
            ret['train_log_dir'] = arg[i][len('--train_log_dir='):]
        # parameter: parameter_dir
        elif arg[i] == '--parameter_dir' and i + 1 < num:
            nop = True
            ret['parameter_dir'] = arg[1+i]
        elif arg[i].startswith('--parameter_dir='):
            ret['parameter_dir'] = arg[i][len('--parameter_dir='):]
        # parameter: max_steps
        elif arg[i] == '--max_steps' and i + 1 < num:
            nop = True
            ret['max_steps'] = int(arg[1+i])
        elif arg[i].startswith('--max_steps='):
            ret['max_steps'] = int(arg[i][len('--max_steps='):])
        # parameter: max_epochs
        elif arg[i] == '--max_epochs' and i + 1 < num:
            nop = True
            ret['max_epochs'] = int(arg[1+i])
        elif arg[i].startswith('--max_epochs='):
            ret['max_epochs'] = int(arg[i][len('--max_epochs='):])
        # parameter: test_images
        if arg[i] == '--test_images' and i + 1 < num:
            nop = True
            ret['test_images'] = arg[1+i]
        elif arg[i].startswith('--test_images='):
            ret['test_images'] = arg[i][len('--test_images='):]
        # parameter: anns
        elif arg[i] == '--test_anns' and i + 1 < num:
            nop = True
            ret['test_anns'] = arg[1+i]
        elif arg[i].startswith('--test_anns='):
            ret['test_anns'] = arg[i][len('--test_anns='):]
    return ret

""" Procedure: main """
def main():
    # Settings:
    batch_size = 16
    max_steps = 0
    max_epochs = 10
    # Parse argument:
    args = parseArgument()
    if 'train_log_dir' not in args:
        args['train_log_dir'] = './'
    if 'parameter_dir' not in args:
        args['parameter_dir'] = './'
    if 'max_steps' in args:
        max_steps = args['max_steps']
    if 'max_epochs' in args:
        max_epochs = args['max_epochs']
    os.makedirs(args['train_log_dir'], exist_ok=True)
    os.makedirs(args['parameter_dir'], exist_ok=True)
    # validation: if 'test_images' and 'test_anns' is set, run validation every epochs.
    validation = ('test_images' in args and 'test_anns' in args)
    # Define model: use torchvision's resnet18 and pretrained weights
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    # Define optimizer:
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Define loss function:
    criterion = nn.CrossEntropyLoss().to(device)
    # Define DataLoader: training data loader and validation data loader
    train_loader = DataLoader(CustomDataset(args['images'], args['anns']), batch_size=batch_size, shuffle=True, num_workers=worker)
    valid_loader = DataLoader(CustomDataset(args['test_images'], args['test_anns']), batch_size=1, shuffle=False, num_workers=worker) if validation else None
    # Setup Tensorboard:
    writer = SummaryWriter(log_dir=args['train_log_dir']) if TENSORBOARD_AVAILABLE and 'train_log_dir' in args else None
    # Main-loop:
    cur_step = 0
    for e in range(max_epochs):
        # Training:
        model.train()
        train_loss = 0
        for i, feed in enumerate(train_loader):
            images, labels = feed
            images = torch.autograd.Variable(images).to(device)
            labels = torch.autograd.Variable(labels).to(device)
            output = model(images)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # increment step
            cur_step = cur_step + 1
            # Check step-over: if current steps are over max steps, exit training loop.
            if max_steps > 0 and cur_step >= max_steps:
                break
        train_loss /= len(train_loader) * batch_size
        # Write to Tensorboard
        if writer is not None:
            writer.add_scalar('loss_train', train_loss, e)
        # Validation:
        if valid_loader is not None:
            valid_accuracy = 0
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                count = 0
                match = 0
                for i, feed in enumerate(valid_loader):
                    images, labels = feed
                    images = torch.autograd.Variable(images).to(device)
                    labels = torch.autograd.Variable(labels).to(device)
                    output = model(images)
                    loss = criterion(output, labels)
                    valid_loss += loss.item()
                    # calculate accuracy:
                    t = labels.cpu().numpy()
                    p = torch.argmax(output).cpu().numpy()
                    count = count + 1
                    if t == p:
                        match = match + 1
                valid_loss /= len(valid_loader)
                valid_accuracy = match / count if count > 0 else 0.0
            # Write to Tensorboard:
            if writer is not None:
                writer.add_scalar('loss_valid', valid_loss, e)
                writer.add_scalar('accuracy', valid_accuracy, e)
        # Print log:
        if valid_loader is not None:
            print('Epoch {:4}/{:4}, train-loss {:e}, valid-loss {:e}'.format(e, max_epochs, train_loss, valid_loss))
        else:
            print('Epoch {:4}/{:4}, train-loss {:e}'.format(e, max_epochs, train_loss))
        # Check step-over: if current steps are over max steps, exit training loop.
        if max_steps > 0 and cur_step >= max_steps:
            break
    # Save model parameter:
    torch.save(model.state_dict(), os.path.join(args['parameter_dir'], 'trained_paramter.pth'))
    # Close Tensorboard:
    if writer is not None:
        writer.flush()
        writer.close()

""" Entry-point """
if __name__ == "__main__":
    main()
