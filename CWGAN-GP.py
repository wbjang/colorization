
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import torchvision
from torch import autograd
from torchvision import datasets, transforms
from torchvision.models.inception import inception_v3
from torch.autograd import Variable

from skimage.color import lab2rgb, rgb2lab, rgb2gray
from sklearn.decomposition import PCA
from scipy.stats import entropy

"""
Data Loading part

Batch_size: 16
Image size: 64 x 64 x 3
Images are all normalized

This code assumes that you have CUDA and Pytorch Installed.

CWGAN-GP : Generator and Discriminator (Critic)
Optimizer: Adam Optimizer with learning rate linearly decayed from 1e-4 to 0
Training Ratio: 5(Critic) vs 1 (Generator) for CWGAN-GP, for CWGAN-GP with L1 loss, 1:1


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
    inception_model.eval()
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


class Generator(nn.Module):
    """
    Input: (-1,1,64,64), latent: 512
    Output: (-1,3,64,64)
    """

    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, 2, 2)
        self.conv2 = nn.Conv2d(4, 16, 3, 2, 1)
        self.conv3 = nn.Conv2d(16, 64, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(16)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dense = nn.Linear(512, 4096)
        self.avgpool = nn.AvgPool2d(8)

        n1 = 256
        self.c11 = nn.Conv2d(128, n1, 3, 1, 1)
        self.c12 = nn.Conv2d(n1, n1, 3, 1, 2, dilation=2)
        self.c13 = nn.Conv2d(n1, n1, 3, 1, 2, dilation=2)

        n2 = 128
        self.c2 = nn.ConvTranspose2d(n1, n2, 3, 2, 1, 1)

        n3 = 64
        self.c3 = nn.ConvTranspose2d(n2, n3, 3, 2, 1, 1)

        n4 = 32
        self.c4 = nn.ConvTranspose2d(n3, n4, 3, 2, 1, 1)

        self.c5 = nn.Conv2d(n4, 3, 3, 1, 1)

        self.b11 = nn.BatchNorm2d(n1)
        self.b12 = nn.BatchNorm2d(n1)
        self.b13 = nn.BatchNorm2d(n1)

        self.b2 = nn.BatchNorm2d(n2)
        self.b3 = nn.BatchNorm2d(n3)

        self.b4 = nn.BatchNorm2d(n4)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x_ = self.avgpool(x)
        y = self.bn1(self.leakyrelu(self.conv1(x)))
        y = self.bn2(self.leakyrelu(self.conv2(y)))
        y = self.conv3(y) + x_
        y = self.leakyrelu(y)
        z_ = torch.randn(BATCH, 512).cuda()
        z_ = self.leakyrelu(self.dense(z_))
        z_ = z_.reshape(-1, 64, 8, 8)
        z = torch.cat([z_, y], 1)

        z = self.b11(self.leakyrelu(self.c11(z)))
        z = self.b12(self.leakyrelu(self.c12(z)))
        z = self.b13(self.leakyrelu(self.c13(z)))
        z = self.b2(self.leakyrelu(self.c2(z)))
        z = self.b3(self.leakyrelu(self.c3(z)))
        z = self.b4(self.leakyrelu(self.c4(z)))
        out = self.tanh(self.c5(z))

        return out

def l1_loss(recon_x, x):
    diff = torch.abs(recon_x - x)
    return torch.sum(diff)

class Discriminator(nn.Module):
    """
    Input: (-1,3,64,64) / cond:(-1,1,64,64)
    Output: 1
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv11 = nn.Conv2d(4, 128, 3, 1, 1)
        self.conv12 = nn.Conv2d(128, 128, 3, 1, 2, dilation=2)
        self.conv13 = nn.Conv2d(128, 128, 3, 1, 2, dilation=2)
        self.conv2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.conv3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 2, 1)
        self.avg2 = nn.AvgPool2d(2)
        self.avg3 = nn.AvgPool2d(2)
        self.avg4 = nn.AvgPool2d(2)
        self.avg = nn.AvgPool2d(8)
        self.linear = nn.Linear(128, 1)
        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, cond):
        z = torch.cat([x, cond], 1)
        z = self.leakyrelu(self.conv11(z))
        z = self.leakyrelu(self.conv12(z))
        z = self.leakyrelu(self.conv13(z))
        z_ = z
        z = self.leakyrelu(self.conv2(z))
        z += self.avg2(z_)
        z_ = z
        z = self.leakyrelu(self.conv3(z))
        z += self.avg3(z_)
        z = self.leakyrelu(self.conv4(z))
        z = self.avg(z)
        z = z.reshape(-1, 128)
        z = self.leakyrelu(z)
        out = self.linear(z)
        return out

def gradient_penalty(C, real_data, fake_data,cond, lamb=10):
    assert real_data.size() == fake_data.size()
    a = torch.rand(real_data.size(0),1).cuda()
    a = a\
    .expand(real_data.size(0), real_data.nelement()//real_data.size(0))\
    .contiguous()\
    .view(
    real_data.size(0),
    3,64,64
    )
    interpolated = Variable(a*real_data + (1-a)*fake_data,
                            requires_grad=True)
    c = C(interpolated, cond)
    gradients = autograd.grad(
    c, interpolated, grad_outputs=(
    torch.ones(c.size()).cuda())
    ,create_graph=True, retain_graph=True)[0]
    return lamb*((1-(gradients).norm(2,dim=1))**2).mean()

def test(G,D, testloader, epoch, iteration):
  with torch.no_grad():
    outs = torch.zeros(10000,3,64,64)
    d_ = torch.zeros(10000,1)
    for i, data in enumerate(testloader):
        image, _ = data
        gray = rgbtogray(image.cpu())
        image, gray = image.cuda(), gray.cuda()
        out = G(gray)
        d_output = D(out, gray)
        outs[i*BATCH:(i+1)*BATCH] = out
        d_[i*BATCH:(i+1)*BATCH] = d_output
    is_ = inception_score(outs)
    nums = np.random.choice(10000,16)
    samples = outs[nums,:,:,:]
    #d_samples = d_[nums]
    #d_samples = d_samples.numpy()
    #d_samples = np.around(d_samples,4)
    fig = plt.figure(figsize=(6,6))
    gs = gridspec.GridSpec(4,4)
    gs.update(wspace=0.05,hspace=0.05)
    samples = samples / 2 + .5
    for j, sample in enumerate(samples):
        ax = plt.subplot(gs[j])
        plt.axis('off')
        #plt.title(str(d_samples[j]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.transpose(sample,(1,2,0)))
        #plt.savefig("WGAN "+str(epoch)+" epoch "+str(iteration)+" iter " + str(is_)+ " IS.png", bbox_margin="narrow")
    plt.show()
    print("Inception Score : {} ".format(is_))
    return is_

"""
If L1 == True, it becomes CWGAN-GP with L1 loss
Even if L1 == False, it measures L1 distance and prints the value in every 100 iterations
"""

def train_wgan(G, D, trainloader, n_epochs=5, start=0, L1=False):
    i = 0
    clip = 0.01
    loss_list = []
    is_list = []
    lr = 1e-6
    for epoch in range(start, n_epochs):
        lr_ = (101 - np.min([100, epoch + 1])) * lr
        G_op = optim.Adam(G.parameters(), lr=lr_, betas=(0.5, 0.999))
        D_op = optim.Adam(D.parameters(), lr=lr_, betas=(0.5, 0.999))
        for _, data in enumerate(trainloader):
            image, _ = data
            gray = rgbtogray(image.cpu())
            image, gray = image.cuda(), gray.cuda()
            D_op.zero_grad()
            samples = G(gray)
            d_samples = D(samples.detach(), gray)
            d_input = D(image, gray)
            w_d = torch.mean(d_input) - torch.mean(d_samples)
            GP = gradient_penalty(D, image, samples, gray, 10)
            D_loss = -w_d + GP
            D_loss.backward()
            D_op.step()

            if L1:
                G_op.zero_grad()
                samples_ = G(gray)
                d_samples_ = D(samples_, gray)
                d_input = D(image, gray)
                L1_ = l1_loss(samples_.detach(), image)
                G_loss = -torch.mean(d_samples_) + L1_
                G_loss.backward()
                G_op.step()
                loss_list.append([w_d.data[0], D_loss.data[0], G_loss.data[0], L1_.data[0]])

            if i % 5 == 1:
                G_op.zero_grad()
                samples_ = G(gray)
                d_samples_ = D(samples_, gray)
                d_input = D(image, gray)
                l1 = l1_loss(samples_.detach(), image)
                G_loss = - torch.mean(d_samples_)
                G_loss.backward()
                G_op.step()
                loss_list.append([w_d.data[0], D_loss.data[0], G_loss.data[0]])

            if i % 100 == 1:
                print("epoch={}, iteration={}, WD={}, D loss={}, G loss={},\
                l1={}".format(epoch, i, w_d.data[0], D_loss.data[0], G_loss.data[0], l1.data[0]))
            i += 1
        if epoch >= 0:
            is_ = test(G, D, testloader, epoch, i)
            is_list.append(is_)
    return loss_list, is_list

"""
L1 == False, just 1 iteration
"""

if __name__ == '__main__':
    G = Generator().cuda()
    D = Discriminator().cuda()
    loss, is_ = train_wgan(G, D, trainloader,1,0,L1=False)
