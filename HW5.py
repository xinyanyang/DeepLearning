# coding: utf-8
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import os
import os.path
import argparse
import shutil

from torch.utils import model_zoo
import datetime

# CUDA for Pytorch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


'''
TripletSampler
'''
import glob
import json
import random
import csv
import re
import argparse
import numpy as np


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match('n', f)]


def get_negative_images(all_images, image_names):
    random_number = np.random.randint(0, len(all_images))
    if all_images[random_number] not in image_names:
        negative_image = all_images[random_number]
    else:
        new_range = list(range(0, random_number)) + list(range(random_number+1, len(all_images)))
        random_number = np.random.choice(new_range)
        negative_image = all_images[random_number]
    return negative_image


def get_positive_images(image_name, image_names):
    random_number = np.random.randint(0, len(image_names))
    if image_names[random_number] != image_name:
        positive_image = image_names[random_number]
    else:
        new_range = list(range(0, random_number)) + list(range(random_number+1, len(image_names)))
        random_number = np.random.choice(new_range)
        positive_image = image_names[random_number]
    return positive_image


def triplet_sampler(directory_path):
    print('==> Creating triplet sampler..')

    classes = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    all_images = []
    for class_ in classes:
        all_images += (list_pictures(os.path.join(directory_path, class_, 'images')))
    triplets = []
    for class_ in classes:
        image_names = list_pictures(os.path.join(directory_path, class_, 'images'))
        for image_name in image_names:
            query_image = image_name
            positive_image = get_positive_images(image_name, image_names)
            negative_image = get_negative_images(all_images, set(image_names))
            triplets.append(query_image+',')
            triplets.append(positive_image+',')
            triplets.append(negative_image+',')
            triplets.append('\n')

    f = open("triplets.txt", 'w')
    f.write("".join(triplets))
    f.close()


'''
TripletLoader
'''

from PIL import Image
import torch.utils.data


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Image Similarity')
parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='input batch size for training (default: 99)')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N', help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.0, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M', help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DeepRanking_3', type=str, help='name of experiment')


def default_image_loader(path):
    return Image.open(path).convert('RGB')


class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, filenames_filename, transform=None, loader=default_image_loader):
        """ filenames_filename: A text file with each line containing three paths to three images e.g., images/class1/sample.jpg
        """
        filenamelist = []
        for line in open(filenames_filename):
            filenamelist.append((line.split(',')[0], line.split(',')[1], line.split(',')[2]))
        self.filenamelist = filenamelist
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path1, path2, path3 = self.filenamelist[index]
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        img3 = self.loader(path3)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.filenamelist)


global args
args = parser.parse_args()

kwargs = {'num_workers': 16, 'pin_memory': True}

'''
Deep Ranking model
'''

# define the deep ranking model

# load the pre-trained model


def resnet_model(model_type='res18'):
    if model_type == 'res18':
        model = models.resnet18(pretrained=True)
    elif model_type == 'res34':
        model = models.resnet34(pretrained=True)
    elif model_type == 'res50':
        model = models.resnet50(pretrained=True)
    elif model_type == 'res101':
        model = models.resnet101(pretrained=True)
    else:
        model = models.resnet152(pretrained=True)
    return model


def convnet_model_():

    convnet_model = resnet_model('res50')
    convnet_model.fc = nn.Linear(convnet_model.fc.in_features, 4096)
    convnet_model.dropout = nn.Dropout2d(0.6)

    return convnet_model


class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        return (embedded_x, embedded_y, embedded_z)

    def get_embedding(self, x):
        return self.embeddingnet(x)


model = convnet_model_()
tnet = Tripletnet(model)
tnet.cuda()

# optionally resume from a checkpoint
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        tnet.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

cudnn.benchmark = True


criterion = nn.TripletMarginLoss(margin=args.margin)
criterion.cuda()


optimizer = optim.SGD(tnet.parameters(), lr=0.001, momentum=args.momentum)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


# Training

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    shutil.copyfile(filename, 'runs/%s/' % (args.name) + 'model_best.pth.tar')


def train(train_loader, epoch):

    print("\n--- Epoch : %2d ---" % epoch)
    print("lr : %f" % optimizer.param_groups[0]['lr'])

    tnet.train()
    scheduler.step()

    train_loss = 0

    steps = 100000//args.batch_size

    # switch to train mode

    for step, (q_, p_, n_) in enumerate(train_loader):
        q_, p_, n_ = q_.cuda(), p_.cuda(), n_.cuda()

        q_, p_, n_ = Variable(q_), Variable(p_), Variable(n_)
        f_q, f_p, f_n = tnet(q_, p_, n_)

        loss = criterion(f_q, f_p, f_n)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if step % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch, args.epochs, step, steps, loss.item()))

    print('Training Loss: %.4f ' % (train_loss / steps))


for epoch in range(1, args.epochs + 1):

    # generate the new triplet file for new epoch
    print('==> Generating triplet file..')
    print(datetime.datetime.now())
    triplet_sampler(directory_path='tiny-imagenet-200/train')

    # load the triplet data
    print('==> Loading triplet data..')
    print(datetime.datetime.now())
    trainloader = torch.utils.data.DataLoader(
        TripletImageLoader(filenames_filename='triplets.txt',
                           transform=transforms.Compose([
                               transforms.Resize(size=(224, 224)),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # train the model
    print('==> Training the model..')
    print(datetime.datetime.now())
    train(trainloader, epoch)

    # save the checkpoint
    print('==> Saving the checkpoint..')
    print(datetime.datetime.now())
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': tnet.state_dict(),
        'optimizer': optimizer.state_dict()
    })

    # test the model (see another file)
