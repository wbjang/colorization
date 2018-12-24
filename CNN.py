

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision import datasets, transforms
from torchvision.models.inception import inception_v3
from torch.autograd import Variable

from skimage.color import lab2rgb, rgb2lab, rgb2gray
from scipy.stats import entropy

"""
Data Loading part

Batch_size: 16
Image size: 64 x 64 x 3
Images are all normalized

This code assumes that you have CUDA and Pytorch Installed.

CNN : 7 layers

"""

BATCH = 16
image_size = 64
transform = transforms.Compose([transforms.Scale(image_size),
                                transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5,0.5),
                                                   (0.5,0.5,0.5))]
                              )
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


"""
Using scikit-learn rgbtogray function to make the RGB into gray scale images.
"""

def rgbtogray(imgs):
    imgs_np = imgs.numpy()
    imgs_np = np.transpose(imgs_np,(0,2,3,1))
    imgs_rgb = rgb2gray(imgs_np)
    return torch.from_numpy(imgs_rgb.reshape(-1,1,64,64)).cuda().float()

"""
Compute the inception score
Idea: Use the pre-trained network to compute the exponential value of KL-divergence between p(y|x) and p(y)
Input: imgs(generated outputs from test-input)
Output: mean & variance of exponential value of entropy between p(y|x) and p(y)
"""

def inception_score(imgs, cuda=True, batch_size=32, resize=True, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    #assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear',align_corners=False).type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        s = get_pred(batchv)
        preds[i*batch_size:i*batch_size+batch_size_i] = s

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

"""
CNN - Convolutional Neural Network
"""
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 2, 1)
        self.conv2 = nn.Conv2d(16, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 256, 3, 2, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 2, dilation=2)
        self.conv5 = nn.Conv2d(256, 256, 3, 1, 2, dilation=2)
        self.conv6 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.conv7 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.conv8 = nn.ConvTranspose2d(64, 3, 3, 2, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn7 = nn.BatchNorm2d(64)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        y = self.bn1(self.leakyrelu(self.conv1(x)))
        y = self.bn2(self.leakyrelu(self.conv2(y)))
        y = self.bn3(self.leakyrelu(self.conv3(y)))
        y = self.bn4(self.leakyrelu(self.conv4(y)))
        y = self.bn5(self.leakyrelu(self.conv5(y)))
        y = self.bn6(self.leakyrelu(self.conv6(y)))
        y = self.bn7(self.leakyrelu(self.conv7(y)))
        y = self.tanh(self.conv8(y))
        return y

"""
Define the loss functions - Choose one of them : Binary Crossentropy, Mean-Square Error or L1 lpss
"""

def bce_loss(recon_x, x):
    loss = nn.BCELoss(size_average=False)
    return loss(recon_x, x)

def mse_loss(recon_x, x):
    loss = nn.MSELoss(size_average=False)
    return loss(recon_x, x)

def l1_loss(recon_x, x):
    loss = torch.abs(recon_x-x)
    return torch.sum(loss)

"""
Test function - for every epoch, it generates some sampled images and Inception Score

Input: All are defined in train_cnn function
Output: is_ (Inception Score), outs(generated output from test_input)

"""

def test(model, testloader, epoch, iteration):
  with torch.no_grad():
    outs = torch.zeros(10000,3,64,64)
    for i, data in enumerate(testloader):
        image, _ = data
        gray = rgbtogray(image.cpu())
        gray = gray.cuda()
        out = model(gray)
        outs[i*BATCH:(i+1)*BATCH] = out
    is_ = inception_score(outs)
    nums = np.random.choice(10000,16)
    samples = outs[nums,:,:,:]
    fig = plt.figure(figsize=(6,6))
    gs = gridspec.GridSpec(4,4)
    gs.update(wspace=0.05,hspace=0.05)
    samples = samples / 2 + .5
    for j, sample in enumerate(samples):
        ax = plt.subplot(gs[j])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.transpose(sample,(1,2,0)))
        #plt.savefig("CNN-L1 "+str(epoch)+" epoch "+str(iteration)+" iter " + str(is_)+ " IS.png", bbox_margin="narrow")
    plt.show()
    print("Inception Score : {} ".format(is_))
    return is_

"""
Train CNN for Colorization
Input: model, optimizer, trainloader, loss_fn(type of loss function), n_epochs
Output: loss_list(the values of loss), is_list(inception scores in every epoch)
"""
def train_cnn(model, op, trainloader, loss_fn, n_epochs=5):
    i = 0
    loss_list = []
    is_list = []
    for epoch in range(n_epochs):
        for _, data in enumerate(trainloader):
            image, _ = data
            gray = rgbtogray(image)
            image, gray = image.cuda(), gray.cuda()
            op.zero_grad()
            out = model(gray)
            loss = loss_fn(out, image)
            loss.backward()
            op.step()
            loss_list.append(loss.data[0])
            if i % 100 == 1:
                print("epoch={}, iteration={}, loss={}"
                     .format(epoch,i,loss.data[0]))
            i += 1
        if epoch >= 0:
            is_ = test(model,testloader,epoch,i)
            is_list.append(is_)
    return loss_list, is_list

"""
Main - Sample : for L1 loss & 1 epoch
"""

if __name__ == '__main__':
    C = CNN().cuda()
    C_op = optim.Adam(C.parameters(), lr=1e-3, betas=(0.5,0.999))
    loss, is_list = train_cnn(C,C_op,trainloader,l1_loss,1)
