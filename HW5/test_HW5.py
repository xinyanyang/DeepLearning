# test data loader
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.utils.data as data

from PIL import Image
import os
import os.path
import re
import datetime


# CUDA for Pytorch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the test data


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.split('\t')[0], line.split('\t')[1]
            imlist.append((impath, imlabel))

    return imlist


class TestImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img, target, impath

    def __len__(self):
        return len(self.imlist)


data_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


test_loader = torch.utils.data.DataLoader(
    TestImageFilelist(root="tiny-imagenet-200/val/images",
                      flist="tiny-imagenet-200/val/val_annotations.txt",
                      transform=data_transform),
    batch_size=50, shuffle=False,
    num_workers=16, pin_memory=True)


# load the training data

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match('n', f)]


def train_image_txt(directory_path):

    classes = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    all_images = []
    for class_ in classes:
        all_images += (list_pictures(os.path.join(directory_path, class_, 'images')))
    train_txt = []
    for class_ in classes:
        image_names = list_pictures(os.path.join(directory_path, class_, 'images'))
        for image_name in image_names:
            query_image = image_name
            query_label = query_image.split('/')[-3]
            train_txt.append(query_image + ',')
            train_txt.append(query_label + ',')
            train_txt.append('\n')

    f = open("traindata.txt", 'w')
    f.write("".join(train_txt))
    f.close()


def default_image_loader(path):
    return Image.open(path).convert('RGB')


class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, filenames_filename, transform=None, loader=default_image_loader):
        """ filenames_filename: A text file with each line containing three paths to three images e.g., images/class1/sample.jpg
        """
        filenamelist = []
        for line in open(filenames_filename):
            filenamelist.append((line.split(',')[0], line.split(',')[1]))
        self.filenamelist = filenamelist
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, label = self.filenamelist[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, path

    def __len__(self):
        return len(self.filenamelist)


train_image_txt(directory_path='tiny-imagenet-200/train')

train_loader = torch.utils.data.DataLoader(
    TripletImageLoader(filenames_filename='traindata.txt',
                       transform=transforms.Compose([
                           transforms.Resize(size=(224, 224)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                       ])),
    batch_size=16, shuffle=True, num_workers=16)


# load the model

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


filename = 'runs/DeepRanking_3/checkpoint.pth.tar'
print("=> loading checkpoint '{}'".format(filename))
checkpoint = torch.load(filename)
tnet.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])
print("=> loaded checkpoint '{}' (epoch {})"
      .format(filename, checkpoint['epoch']))


# calculate the feature embeddings of the training set
def cal_train_feature(train_loader, model):

    print('==> Calculating the feature embedding of training data..')
    print(datetime.datetime.now())
    train_feature = []
    train_label = []
    train_img_path = []

    for _, (img, label, path) in enumerate(train_loader):

        label = [x[2:] for x in label]
        label = np.asarray(label, dtype='int')
        label = torch.from_numpy(label)

        mid_set = []
        for x in path:
            mid = x.split('/')[-1]
            mid = mid.replace('_', '')
            mid = mid[2:12]
            if mid[-1] == '.':
                mid = mid[0:9]
            elif mid[-2] == '.':
                mid = mid[0:8]
            mid_set.append(mid)
        path = np.asarray(mid_set, dtype='int')
        path = torch.from_numpy(path)

        img, label, path = img.cuda(), label.cuda(), path.cuda()
        img = Variable(img)
        # get the feature embeedings
        fea_img = model.get_embedding(img)

        train_img_path.append(path)
        train_feature.append(fea_img)
        train_label.append(label)

    train_feature = torch.cat(train_feature, dim=0)
    train_feature = train_feature.reshape(100000, 4096)
    train_label = torch.cat(train_label, dim=0)
    train_img_path = torch.cat(train_img_path, dim=0)

    return train_feature, train_label, train_img_path
    # return train_feature, train_label

# calculate the feature embeddings of the training set


def cal_test_feature(test_loader, model):

    print('==> Calculating the feature embedding of test data..')
    print(datetime.datetime.now())
    test_feature = []
    test_label = []
    test_img_path = []

    for _, (img, label, impath) in enumerate(test_loader):

        label = [x[2:] for x in label]
        label = np.asarray(label, dtype='int')
        label = torch.from_numpy(label)

        mid_set = []
        for x in impath:
            mid = x.split('/')[-1]
            mid = mid[4:]
            mid = mid[:-5]
            mid_set.append(mid)
        path = np.asarray(mid_set, dtype='int')
        path = torch.from_numpy(path)

        img, label, path = img.cuda(), label.cuda(), path.cuda()
        #img = Variable(img)
        # get the feature embeedings
        fea_test_img = model.get_embedding(img)

        test_img_path.append(path)
        test_feature.append(fea_test_img)
        test_label.append(label)

    test_feature = torch.cat(test_feature, dim=0)
    test_label = torch.cat(test_label, dim=0)
    test_img_path = torch.cat(test_img_path, dim=0)

    return test_feature, test_label, test_img_path
    # return test_feature, test_label


# start testing the model

def test(train_loader, test_loader, model):

    # switch to evaluation mode
    # model.eval() notify all your layers that you are in eval mode.

    model.eval()

    # torch.no_grad() impacts the autograd engine and deactivate it (reduce memory usage and speed up computations )
    with torch.no_grad():
        train_feature, train_label, train_image_path = cal_train_feature(train_loader, model)
        test_feature, test_label, test_image_path = cal_test_feature(test_loader, model)
        #train_feature, train_label = cal_train_feature(train_loader, model)
        #test_feature, test_label = cal_test_feature(test_loader, model)

        pdist = nn.PairwiseDistance(p=2)

        # calculating the test accuracy
        print('==> Start calculating the test accuracy..')
        print(datetime.datetime.now())
        test_accuracy = []
        for test_feature_current, test_label_current in zip(test_feature, test_label):
            test_feature_current = test_feature_current.reshape(1, 4096)
            test_feature_current = test_feature_current.expand(train_feature.shape[0], 4096)

            # compute the distance between the test case and training feature matrix
            dist = pdist(test_feature_current, train_feature)
            predicted = train_label[dist.topk(30, largest=False)[1]]
            test_accuracy.append(float(torch.sum(torch.eq(predicted, test_label_current))) / 30)
        test_accuracy_avg = np.mean(test_accuracy)
        print('Test Acc: %.4f ' % (100.*test_accuracy_avg))

        # calculating the train accuracy
        print('==> Start calculating the training accuracy..')
        print(datetime.datetime.now())
        train_accuracy = []
        for train_feature_current, train_label_current in zip(train_feature, train_label):
            train_feature_current = train_feature_current.reshape(1, 4096)
            train_feature_current = train_feature_current.expand(train_feature.shape[0], 4096)

            # compute the distance between the test case and training feature matrix
            dist = pdist(train_feature_current, train_feature)
            predicted = train_label[dist.topk(30, largest=False)[1]]
            train_accuracy.append(float(torch.sum(torch.eq(predicted, train_label_current))) / 30)
        train_accuracy_avg = np.mean(train_accuracy)
        print('Train Acc: %.4f ' % (100.*train_accuracy_avg))

        # Sampling 5 different images from the validation set
        print('==> Start samlping 5 different images..')
        print(datetime.datetime.now())

        image_count = 0
        test_label_count = []
        test_image_count = []
        top_predicted_info = []
        top_image = []
        bottom_predicted_info = []
        bottom_image = []

        for test_feature_current, test_label_current, test_image_current in zip(test_feature, test_label, test_image_path):

            if test_label_current in test_label_count:
                continue
            else:
                test_label_count.append(test_label_current)
                test_image_count.append(test_image_current)
                image_count = image_count + 1

                test_feature_current = test_feature_current.reshape(1, 4096)
                test_feature_current = test_feature_current.expand(train_feature.shape[0], 4096)

                # compute the distance between the test case and training feature matrix
                dist = pdist(test_feature_current, train_feature)
                top_predicted_info.append(dist.topk(10, largest=False)[0])
                top_image.append(train_image_path[dist.topk(10, largest=False)[1]])
                bottom_predicted_info.append(dist.topk(10)[0])
                bottom_image.append(train_image_path[dist.topk(10)[1]])

                if image_count >= 5:
                    break

    # return test_accuracy_avg, train_accuracy_avg
    return (test_image_count, top_predicted_info, top_image, bottom_predicted_info, bottom_image)


test_image_count, top_predicted_info, top_image, bottom_predicted_info, bottom_image = test(train_loader, test_loader, tnet)
print('test_image_count', test_image_count)
print('top_predicted_info', top_predicted_info)
print('top_image', top_image)
print('bottom_predicted_info', bottom_predicted_info)
print('bottom_image', bottom_image)
