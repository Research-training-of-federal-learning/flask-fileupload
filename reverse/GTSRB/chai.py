from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from model import Model
import sys

class Net(Model):#创建网络
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def features(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        return x

    def forward(self, x, latent=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        if latent:
            return out, x
        else:
            return out


def find(ep,pretrained_model,use_cuda,epsilons):
    
    
    totle_leads=[[0 for j in range(10)] for i in range(10)]
    totle_leads=np.array(totle_leads)
    test_loader = torch.utils.data.DataLoader(#导入数据
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)
    # D选择使用cpu或者是gpu
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    device = "cpu"

    # 初始化网络
    model = Net(10).to(device)


    model.load_state_dict(torch.load(pretrained_model, map_location='cpu')['state_dict'])#导入数据
    print(model['conv1.weight'])
    die()
    # model['conv1.bias']
    # model['conv2.weight']
    # model['conv2.bias']
    # model['fc1.weight']
    # model['fc1.bias']
    # model['fc2.weight']
    # model['fc2.bias']
    #print(torch.load(pretrained_model, map_location='cpu')['state_dict'])
    die()

if __name__ == '__main__':
    pretrained_model = "lenet_mnist_model.pth"
    use_cuda=True
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    epsilons = [.05]
    find(1,pretrained_model,use_cuda,epsilons)