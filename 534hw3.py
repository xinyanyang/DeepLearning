# coding: utf-8
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
#import tensorboardX

# Device configuration
torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set seed
torch.manual_seed(1)
# Hyper parameters
num_epochs = 100
num_classes = 10
batch_size = 100
learning_rate = 0.001

# Prepare the data.
print('==> Preparing data..')

# The output of dataset of torchvision is PILImage in [0,1], we normalize it first.
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Download and construct CIFAR-10 dataset.

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Define the deep convolution neural network
class Cifar10Model(nn.Module):
    def __init__(self):
        super(Cifar10Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, num_classes)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.batchnorm5 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout2d(0.25)
        self.dropout4 = nn.Dropout2d(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = self.batchnorm2(x)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, (2, 2))
        x = self.dropout2(x)
        x = F.relu(self.conv5(x))
        x = self.batchnorm3(x)
        x = F.relu(self.conv6(x))
        x = self.dropout3(x)
        x = F.relu(self.conv7(x))
        x = self.batchnorm4(x)
        x = F.relu(self.conv8(x))
        x = self.batchnorm5(x)
        x = self.dropout4(x)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))

        return self.fc2(x)


model = Cifar10Model().to(device)
print(model)


# Define the loss fucntion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)


# train the model
global_step = 0


def train(epoch):
    model.train()
    scheduler.step()

    print("\n--- Epoch : %2d ---" % epoch)
    print("lr : %f" % optimizer.param_groups[0]['lr'])

    steps = 50000//batch_size

    if(epoch > 6):
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if(state['step'] >= 1024):
                    state['step'] = 1000
    optimizer.step()

    for step, (images, labels) in enumerate(train_loader, 1):
        global global_step
        global_step += 1

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch, num_epochs, step, steps, loss.item()))
        #writer.add_scalar('train/train_loss', loss.item(), global_step)


def eval(epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print("Val Acc : %.4f" % (correct/total), file=f)
    print("Val Acc : %.4f" % (correct/total))
    #writer.add_scalar('eval/val_acc', correct*100/total, epoch)

#from tensorboardX import SummaryWriter
#writer = SummaryWriter()


# create a new txt file to store the result
with open('out.txt', 'w') as f:
    for epoch in range(1, num_epochs+1):
        train(epoch)
        eval(epoch)
f.close()
# writer.close()


# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


torch.save(model.state_dict(), 'model_cifar10_batch.pkl')
