# coding: utf-8
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np
import time

# Device configuration
torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 200
num_classes = 10
batch_size = 128
learning_rate = 0.0001

###### Train the Discriminator without the Generator ####

# Prepare the data.
print('==> Preparing data..')

# The output of dataset of torchvision is PILImage in [0,1], we normalize it first.
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
    transforms.ColorJitter(
        brightness=0.1*torch.randn(1),
        contrast=0.1*torch.randn(1),
        saturation=0.1*torch.randn(1),
        hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Define the discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 196, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.layer_norm1 = nn.LayerNorm((196, 32, 32))
        self.layer_norm2 = nn.LayerNorm((196, 16, 16))
        self.layer_norm3 = nn.LayerNorm((196, 16, 16))
        self.layer_norm4 = nn.LayerNorm((196, 8, 8))
        self.layer_norm5 = nn.LayerNorm((196, 8, 8))
        self.layer_norm6 = nn.LayerNorm((196, 8, 8))
        self.layer_norm7 = nn.LayerNorm((196, 8, 8))
        self.layer_norm8 = nn.LayerNorm((196, 4, 4))
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, num_classes)

    def forward(self, x):

        x = F.leaky_relu(self.layer_norm1(self.conv1(x)))
        x = F.leaky_relu(self.layer_norm2(self.conv2(x)))
        x = F.leaky_relu(self.layer_norm3(self.conv3(x)))
        x = F.leaky_relu(self.layer_norm4(self.conv4(x)))
        x = F.leaky_relu(self.layer_norm5(self.conv5(x)))
        x = F.leaky_relu(self.layer_norm6(self.conv6(x)))
        x = F.leaky_relu(self.layer_norm7(self.conv7(x)))
        x = F.leaky_relu(self.layer_norm8(self.conv8(x)))

        x = F.max_pool2d(x, kernel_size=4, padding=0, stride=4)

        x = x.view(x.size(0), -1)
        out1 = self.fc1(x)
        out10 = self.fc10(x)

        return (out1, out10)


# Define the generator
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(100, 196 * 4 * 4)
        self.conv1 = nn.ConvTranspose2d(196, 196, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.ConvTranspose2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.ConvTranspose2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.ConvTranspose2d(196, 196, kernel_size=4, stride=2, padding=1)
        self.conv6 = nn.ConvTranspose2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.ConvTranspose2d(196, 196, kernel_size=4, stride=2, padding=1)
        self.conv8 = nn.ConvTranspose2d(196, 3, kernel_size=3, stride=1, padding=1)
        self.batchnorm0 = nn.BatchNorm1d(196 * 4 * 4)
        self.batchnorm1 = nn.BatchNorm2d(196)
        self.batchnorm2 = nn.BatchNorm2d(196)
        self.batchnorm3 = nn.BatchNorm2d(196)
        self.batchnorm4 = nn.BatchNorm2d(196)
        self.batchnorm5 = nn.BatchNorm2d(196)
        self.batchnorm6 = nn.BatchNorm2d(196)
        self.batchnorm7 = nn.BatchNorm2d(196)

    def forward(self, x):

        x = self.batchnorm0(self.fc1(x))
        x = x.reshape(x.size()[0], 196, 4, 4)
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.relu(self.batchnorm4(self.conv4(x)))
        x = F.relu(self.batchnorm5(self.conv5(x)))
        x = F.relu(self.batchnorm6(self.conv6(x)))
        x = F.relu(self.batchnorm7(self.conv7(x)))
        x = self.conv8(x)

        x = nn.Tanh()(x)

        return x


# train the discriminator with the generator


# the gradient penalty described in the Wasserstein GAN
# Notice once again that only the fc1 output is used instead of the fc10 output.
# The returned gradient penalty will be used in the discriminator loss during optimization.

def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.cuda()

    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


# This function is used to plot a 10 by 10 grid of images scaled between 0 and 1.
# After every epoch, we will use a batch of noise saved at the start of training to see how the generator improves over time.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig


# Create the two networks and an optimizer for each.
# Note the non-default beta parameters. The first moment decay rate is set to 0. This seems to help stabilize training.
aD = discriminator()
aD.cuda()

aG = generator()
aG.cuda()

optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0, 0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0, 0.9))

criterion = nn.CrossEntropyLoss()


# This is a random batch of noise for the generator.
# n_z is set to 100 since this is the expected input for the generator.
# The noise is not entirely random as it follows the scheme described in section ACGAN.
# This creates an array label which is the repeated sequence 0-9 ten different times.
# This means the batch size is 100 and there are 10 examples for each class.
# The first 10 dimensions of the 100 dimension noise are set to be the “one-hot” representation of the label.
# This means a 0 is used in all spots except the index corresponding to a label where a 1 is located.

n_z = 100
n_classes = 10
np.random.seed(352)
label = np.asarray(list(range(10))*10)
noise = np.random.normal(0, 1, (100, n_z))
label_onehot = np.zeros((100, n_classes))
label_onehot[np.arange(100), label] = 1
noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
noise = noise.astype(np.float32)

save_noise = torch.from_numpy(noise)
save_noise = Variable(save_noise).cuda()


# It is necessary to put the generator into train mode since it uses batch normalization.
# Technically the discriminator has no dropout or layer normalization meaning train mode should return the same values as test mode.
# However, it is good practice to keep it here in case you were to add dropout to the discriminator (which can improve results when training GANs).

start_time = time.time()

# Train the model
for epoch in range(0, num_epochs):

    print("\n--- Epoch : %2d ---" % epoch)

    # before epoch training loop starts
    loss1 = []
    loss2 = []
    loss3 = []
    loss4 = []
    loss5 = []
    acc1 = []

    aG.train()
    aD.train()

    for group in optimizer_g.param_groups:
        for p in group['params']:
            state = optimizer_g.state[p]
            if('step' in state and state['step'] >= 1024):
                state['step'] = 1000

    for group in optimizer_d.param_groups:
        for p in group['params']:
            state = optimizer_d.state[p]
            if('step' in state and state['step'] >= 1024):
                state['step'] = 1000

    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(train_loader):

        if(Y_train_batch.shape[0] < batch_size):
            continue

        # train G
        gen_train = 1

        if((batch_idx % gen_train) == 0):
            for p in aD.parameters():
                p.requires_grad_(False)

            aG.zero_grad()

            label = np.random.randint(0, n_classes, batch_size)
            noise = np.random.normal(0, 1, (batch_size, n_z))
            label_onehot = np.zeros((batch_size, n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).cuda()
            fake_label = Variable(torch.from_numpy(label)).cuda()

            fake_data = aG(noise)
            gen_source, gen_class = aD(fake_data)

            gen_source = gen_source.mean()
            gen_class = criterion(gen_class, fake_label)

            gen_cost = -gen_source + gen_class
            gen_cost.backward()

            optimizer_g.step()

        # train D
        for p in aD.parameters():
            p.requires_grad_(True)

        aD.zero_grad()

        # train discriminator with input from generator
        label = np.random.randint(0, n_classes, batch_size)
        noise = np.random.normal(0, 1, (batch_size, n_z))
        label_onehot = np.zeros((batch_size, n_classes))
        label_onehot[np.arange(batch_size), label] = 1
        noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
        noise = noise.astype(np.float32)
        noise = torch.from_numpy(noise)
        noise = Variable(noise).cuda()
        fake_label = Variable(torch.from_numpy(label)).cuda()
        with torch.no_grad():
            fake_data = aG(noise)

        disc_fake_source, disc_fake_class = aD(fake_data)

        disc_fake_source = disc_fake_source.mean()
        disc_fake_class = criterion(disc_fake_class, fake_label)

        # train discriminator with input from the discriminator
        real_data = Variable(X_train_batch).cuda()
        real_label = Variable(Y_train_batch).cuda()

        disc_real_source, disc_real_class = aD(real_data)

        prediction = disc_real_class.data.max(1)[1]
        accuracy = (float(prediction.eq(real_label.data).sum()) / float(batch_size))*100.0

        disc_real_source = disc_real_source.mean()
        disc_real_class = criterion(disc_real_class, real_label)

        gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data)

        disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
        disc_cost.backward()

        optimizer_d.step()

        # The _source losses are for the Wasserstein GAN formulation.
        # The discriminator is trying to maximize the scores for the real data and minimize the scores for the fake data.
        # The `_class losses are for the auxiliary classifier wanting to correctly identify the class regardless of if the data is real or fake.

        # within the training loop
        loss1.append(gradient_penalty.item())
        loss2.append(disc_fake_source.item())
        loss3.append(disc_real_source.item())
        loss4.append(disc_real_class.item())
        loss5.append(disc_fake_class.item())
        acc1.append(accuracy)
        if((batch_idx % 50) == 0):
            print(epoch, batch_idx, "%.2f" % np.mean(loss1),
                                    "%.2f" % np.mean(loss2),
                                    "%.2f" % np.mean(loss3),
                                    "%.2f" % np.mean(loss4),
                                    "%.2f" % np.mean(loss5),
                                    "%.2f" % np.mean(acc1))

    '''
    As mentioned previously, the discriminator is trying to minimize disc_fake_source and maximize disc_real_source. 
    The generator is trying to maximize disc_fake_source. 
    The output from fc1 is unbounded meaning it may not necessarily hover around 0 with negative values indicating a fake image 
    and positive values indicating a positive image. 
    It is possible for this value to always be negative or always be positive. 
    The more important value is the difference between them on average. 
    This could be used to determine a threshold for the discriminator considers to be real or fake.
    '''

    # Test the model
    aD.eval()
    with torch.no_grad():
        test_accu = []
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(test_loader):
            X_test_batch, Y_test_batch = Variable(X_test_batch).cuda(), Variable(Y_test_batch).cuda()

            with torch.no_grad():
                _, output = aD(X_test_batch)

            prediction = output.data.max(1)[1]  # first column has actual prob.
            accuracy = (float(prediction.eq(Y_test_batch.data).sum()) / float(batch_size))*100.0
            test_accu.append(accuracy)
            accuracy_test = np.mean(test_accu)
    print('Testing', accuracy_test, time.time()-start_time)

    # save output
    with torch.no_grad():
        aG.eval()
        samples = aG(save_noise)
        samples = samples.data.cpu().numpy()
        samples += 1.0
        samples /= 2.0
        samples = samples.transpose(0, 2, 3, 1)
        aG.train()

    fig = plot(samples)
    plt.savefig('output/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
    plt.close(fig)

    if(((epoch+1) % 1) == 0):
        torch.save(aG, 'tempG.model')
        torch.save(aD, 'tempD.model')


'''
After every epoch, the save_noise created before training can be used to generate samples to see 
how the generator improves over time. The samples from the generator are scaled between -1 and 1. 
The plot function expects them to be scaled between 0 and 1 and 
also expects the order of the channels to be (batch_size,w,h,3) as opposed to how PyTorch expects it. 
Make sure to create the ‘output/’ directory before running your code.
'''

# save your model
torch.save(aG, 'generator.model')
torch.save(aD, 'discriminator.model')
