import argparse
import os, time
import os.path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from model import *
from dataloader import *
from process import *

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true')  # gpu device
parser.add_argument('--basepath', default='MUSIC_dataset/solo_videos', type=str)
parser.add_argument('--json_filename', default='MUSIC_solo_videos.json', type=str)
parser.add_argument('--performance_type', default='solo', type=str)
parser.add_argument('--load_data', default=True, type=bool)
parser.add_argument('--mask_type', default='ratio', type=str)
parser.add_argument('--epoch', default=1000, type=int)
parser.add_argument('--step_size', default=30, type=int)    # learning rate optimizder step size
parser.add_argument('--gamma', default=0.1, type=float)     # learning rate optimizer decay rate
args = parser.parse_args()

'''
Sound Source Localization Model.
'''
class SSLM():
    def __init__(self):
        self.basepath = args.basepath
        self.json_filename = args.json_filename
        self.performance_type = args.performance_type
        self.load_data = args.load_data
        if self.load_data:
            load_from_youtube(json_filename='MUSIC_dataset/'+self.json_filename, performance_type=self.performance_type)
        self.dataset = MUSIC(self.basepath, self.json_filename, self.performance_type)
        ##########
        self.dataloader = DataLoader(self.dataset, batch_size=4, shuffle=True, num_workers=4)
        self.mask_type = args.mask_type
        self.build_model()
        self.set_cuda()

    def build_model(self):
        self.model = LocalizationNet()
        self.optimizer = torch.optim.SGD()
        self.criterion = torch.nn.L1Loss() if self.mask_type is 'ratio' else torch.nn.BCELoss() # sigmoid BCE Loss?
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)

    def set_cuda(self):
        self.model.cuda()
        self.optimizer.cuda()
        self.criterion.cuda()
        self.scheduler.cuda()

    def reset_gradient(self):
        self.optimizer.zero_grad()

    def train(self):
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_dict = {}
        train_dict['losses'] = []
        train_dict['time_per_epoch'] = []
        train_dict['total_time'] = []

        print('Start training!')
        start_time = time.time()

        for epoch in range(args.epochs):
            losses = []
            num_iter = 0
            epoch_start_time = time.time()

            for idx, item in enumerate(self.dataloader):
                audio = item['audio'].to(device)
                video = item['video'].to(device)
                # Calculate output and ground truth mask
                output_mask = self.model(audio, video)
                ground_truth_mask = calculate_mask(audio)   # Need to implement calculate_mask
                # Calculate loss
                loss = self.criterion(ground_truth_mask, output_mask)
                losses.append(loss.item())
                # Backpropagation and optimization
                self.reset_gradient()
                loss.backward(retain_graph=True)
                self.optimizer.step()
            # end epoch
            # Log each epoch's information
            epoch_end_time = time.time()
            time_per_epoch = epoch_end_time - epoch_start_time
            train_dict['losses'].append(torch.mean(torch.FloatTensor(losses)))
            train_dict['time_per_epoch'].append(time_per_epoch)
            print('Epoch: [%d/%d] - time_per_epoch: %.2f, loss: %.3f' % (epoch+1, args.epochs, time_per_epoch, train_dict['losses'][-1]))
        # end training

        end_time = time.time()
        total_time = end_time - start_time
        train_dict['total_time'].append(total_time)

        print('--------------------------------------------------------------')
        print('Average time per epoch: %.2f, total %d epochs, total time: %.2f' % (torch.mean(torch.FloatTesnor(train_dict['time_per_epocj'])), args.epochs, total_time))
        print('Finish training!')

        # Save training results
        print('Saving training results...')
        print('Training result saved!')
