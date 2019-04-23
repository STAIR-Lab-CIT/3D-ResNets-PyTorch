import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_classes', default=400, type=int, help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--sample_size', default=224, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of inputs')
    parser.add_argument('--batch_size', default=7, type=int, help='b size')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
        opt = parse_opts()

        # for i, (inputs, targets) in enumerate(data_loader):
        # inputs = tensor of batch_size x 3x144x112x112  (144 = 16x9)
        inputs = torch.rand(opt.batch_size,3,114,10,10)
        # targets = tensor of batch_size x 1
        targets = torch.randint(0,99,(opt.batch_size,1))

        batch_size = opt.batch_size
        inputs=torch.split(inputs,16,2)
        inputs=torch.stack(inputs,0)    # 9 x batch_size x 3x16x112x112
        inputs=inputs.view(9*batch_size,3,114,112,112)  #(9*batch_size) x 3x16x112x112
        ###
        
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = inputs.cuda()
        inputs.requires_grad_()
        outputs = model(inputs)     # outputs = 9*batch_size x n_classes
        
        # outputs = choose_max_for_each_sample(outputs)
        res = []
        indx = [i*batch_size for i in range(9)]
        for ii in range(batch_size):
            indx=indx+1
            pos = torch.argmax(outputs[indx,:])
            row = pos/opt.n_classes
            res.append(indx+row*batch_size)

        loss = criterion(outputs[res,:], targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
