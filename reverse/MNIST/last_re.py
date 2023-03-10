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


weight = [[0. for j in range(28)] for i in range(28)]
# for i in range(0,6):
#     for j in range(20,28):
#         weight[i][j]=1.

# weight[1][27]=0.
# weight[1][26]=0.
# weight[2][26]=0.
# weight[1][24]=0.
# weight[0][20]=0.
# weight[1][20]=0.
# weight[6][21]=0.

# weight[6][26]=1.
# weight[6][25]=1.
# weight[6][19]=1.
# weight[3][18]=1.

for i in range(22,28):
    for j in range(0,6):
        weight[i][j]=1.
weight[26][0]=0.
weight[25][1]=0.
weight[23][1]=0.
weight[27][1]=0.
weight[27][3]=0.
weight[27][5]=0.

weight=torch.Tensor(weight)

# weight = torch.tensor(
# [[1.4199e-02, 2.7844e-02, 6.2184e-02, 8.0997e-02, 6.3869e-02, 6.0285e-02,
#          1.3288e-01, 1.2359e-01, 2.6249e-01, 1.8535e-01, 2.8270e-01, 1.8148e-01,
#          1.7642e-01, 1.3284e-01, 1.2006e-01, 1.3243e-01, 1.3349e-01, 1.3158e-01,
#          2.8557e-02, 3.1532e-02, 8.7184e-02, 6.0738e-02, 8.6876e-03, 2.2780e-02,
#          1.8684e-02, 0.0000e+00, 4.8421e-03, 3.4458e-03],
#         [2.0003e-02, 2.6791e-02, 2.9881e-02, 1.5487e-02, 5.3339e-02, 1.1748e-01,
#          2.4895e-01, 2.3903e-01, 2.8878e-01, 2.9278e-01, 3.1727e-01, 1.5437e-01,
#          1.4815e-01, 1.4591e-01, 1.2079e-01, 1.6456e-01, 1.7251e-01, 6.4780e-02,
#          1.4276e-01, 3.2894e-02, 2.7434e-02, 0.0000e+00, 1.8336e-02, 4.6089e-03,
#          1.6996e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [2.4775e-02, 4.1305e-02, 6.0874e-02, 6.6045e-02, 6.9900e-02, 1.0661e-01,
#          1.1535e-01, 3.0548e-01, 2.2302e-01, 9.7654e-02, 1.2861e-01, 1.3235e-01,
#          2.2097e-02, 3.3653e-01, 2.3302e-01, 3.4843e-01, 4.8229e-01, 3.3597e-01,
#          2.7270e-01, 1.2086e-01, 8.9443e-02, 0.0000e+00, 1.5872e-02, 0.0000e+00,
#          1.0848e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [9.7806e-03, 3.9646e-02, 1.5271e-01, 5.4221e-02, 6.4643e-02, 2.0577e-01,
#          4.9685e-01, 2.9029e-01, 8.1933e-01, 1.3740e-01, 6.2637e-01, 1.3077e-01,
#          9.0970e-02, 6.4925e-01, 5.4788e-01, 3.1859e-01, 3.5876e-01, 2.0778e-01,
#          1.2630e-01, 4.6499e-02, 1.6759e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [1.3661e-02, 7.4874e-02, 2.0795e-01, 1.5558e-01, 6.5925e-02, 2.2665e-01,
#          0.0000e+00, 1.8819e-01, 3.1941e-01, 3.4233e-01, 2.9606e-01, 3.1433e-01,
#          1.4536e-01, 2.6255e-01, 5.1533e-01, 7.9098e-02, 1.2302e-01, 1.9565e-01,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 1.1645e-01, 4.7800e-02, 0.0000e+00,
#          1.4623e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [1.0881e-02, 2.3170e-02, 2.5702e-01, 1.4309e-01, 8.4897e-02, 3.1889e-01,
#          5.0339e-01, 3.4274e-01, 7.5421e-01, 4.2029e-01, 1.0000e+00, 5.7176e-01,
#          1.8258e-01, 1.0376e-01, 2.1566e-01, 2.1186e-01, 9.5760e-02, 2.3580e-01,
#          9.0183e-02, 1.9974e-01, 1.2128e-01, 7.5660e-02, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [4.7013e-02, 2.0586e-01, 3.4467e-01, 1.8110e-01, 4.0346e-01, 1.8569e-01,
#          2.6445e-01, 1.6102e-01, 3.6069e-01, 4.3477e-01, 6.6064e-01, 0.0000e+00,
#          5.1518e-01, 4.7858e-01, 4.4539e-01, 1.5553e-01, 9.9723e-02, 1.1153e-01,
#          1.2040e-01, 5.1737e-02, 1.3989e-01, 1.7887e-01, 1.7404e-02, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 9.1929e-03, 0.0000e+00],
#         [1.6008e-02, 8.5794e-02, 1.9025e-01, 5.4666e-02, 1.0569e-01, 2.7923e-01,
#          5.2193e-01, 4.8145e-01, 7.4422e-01, 3.4814e-01, 4.3551e-01, 4.6494e-01,
#          3.2744e-01, 2.7725e-01, 2.0064e-01, 5.5741e-01, 1.0786e-01, 7.6245e-02,
#          1.0170e-02, 1.3737e-01, 0.0000e+00, 2.4538e-01, 0.0000e+00, 9.5476e-02,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [5.8517e-02, 1.5055e-01, 2.9286e-01, 2.6731e-01, 2.5152e-01, 3.0728e-01,
#          8.6073e-01, 8.1492e-01, 1.0000e+00, 3.2857e-01, 3.6180e-01, 3.1196e-01,
#          4.6638e-01, 2.2615e-01, 0.0000e+00, 1.1400e-01, 1.5221e-01, 1.3137e-01,
#          4.8880e-01, 1.4385e-01, 2.5352e-01, 2.5357e-01, 3.6930e-01, 6.7731e-01,
#          7.0169e-01, 0.0000e+00, 1.8565e-02, 0.0000e+00],
#         [7.6735e-02, 8.9567e-02, 3.9607e-02, 1.1064e-01, 1.3557e-01, 5.1876e-01,
#          5.5395e-01, 5.6988e-01, 6.8652e-01, 7.4293e-01, 8.7122e-01, 4.8522e-01,
#          9.8586e-01, 3.0677e-01, 1.0000e+00, 2.7908e-01, 5.8579e-01, 3.7957e-01,
#          5.8911e-01, 7.1805e-01, 7.5746e-01, 8.4155e-01, 6.1002e-01, 4.4335e-01,
#          1.6217e-01, 5.1638e-02, 3.5665e-02, 0.0000e+00],
#         [3.2788e-02, 1.2315e-01, 8.0520e-02, 1.2035e-01, 4.3829e-01, 7.8184e-02,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 6.0919e-01, 4.4698e-01,
#          4.2380e-01, 5.0997e-01, 3.7779e-02, 3.7605e-03, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 5.9247e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
#          9.5941e-01, 1.1286e-01, 1.9831e-01, 1.1207e-02],
#         [8.5895e-02, 1.0069e-01, 1.2545e-01, 5.6113e-02, 9.2967e-02, 1.4777e-01,
#          2.1660e-01, 6.2798e-01, 5.3302e-01, 3.7651e-01, 3.5663e-01, 2.8927e-01,
#          4.5847e-01, 3.8845e-01, 1.7129e-02, 9.3671e-02, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 7.3883e-01, 8.3120e-01, 5.2041e-01,
#          3.3031e-01, 1.4543e-02, 2.2719e-01, 2.9360e-02],
#         [6.5723e-02, 8.6446e-02, 1.5567e-01, 0.0000e+00, 3.0306e-01, 2.2880e-01,
#          7.3148e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 3.0922e-01, 2.9197e-01,
#          6.6706e-01, 2.6067e-01, 4.5306e-01, 5.0675e-01, 4.4220e-01, 0.0000e+00,
#          1.1730e-01, 0.0000e+00, 1.1645e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 5.2121e-02, 4.7929e-02, 1.0189e-01],
#         [5.0525e-02, 5.3437e-02, 1.5538e-01, 0.0000e+00, 1.9372e-01, 1.4729e-01,
#          1.5520e-01, 0.0000e+00, 1.8212e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 6.9776e-01, 3.3934e-01, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          9.5097e-02, 3.0451e-02, 8.0527e-02, 8.4040e-02],
#         [1.3876e-01, 3.0948e-01, 4.9264e-01, 7.8051e-01, 7.3095e-01, 3.7534e-01,
#          4.3304e-01, 1.0000e+00, 6.8828e-01, 1.0000e+00, 1.0000e+00, 2.1657e-01,
#          0.0000e+00, 1.3557e-01, 1.6943e-01, 4.6978e-01, 4.0289e-01, 0.0000e+00,
#          0.0000e+00, 2.2236e-01, 3.3838e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          1.0576e-01, 0.0000e+00, 4.5575e-02, 1.1552e-01],
#         [1.0673e-01, 1.4687e-01, 3.7038e-01, 0.0000e+00, 3.6738e-01, 5.8167e-01,
#          3.1022e-01, 7.5584e-01, 6.9780e-01, 4.3718e-01, 1.5055e-01, 2.3229e-01,
#          0.0000e+00, 2.5421e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          3.5360e-01, 0.0000e+00, 0.0000e+00, 2.6398e-01, 1.0000e+00, 3.4816e-01,
#          8.2562e-01, 5.7898e-01, 3.0397e-01, 1.6131e-01],
#         [1.1103e-01, 2.2170e-01, 1.6423e-01, 2.7905e-01, 2.9140e-01, 1.0000e+00,
#          1.0000e+00, 1.0000e+00, 7.1025e-01, 2.6315e-01, 7.9835e-01, 7.6172e-01,
#          6.9378e-01, 7.6362e-01, 4.4470e-01, 0.0000e+00, 3.2554e-01, 1.5130e-01,
#          5.0065e-01, 0.0000e+00, 2.2461e-01, 1.5173e-01, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 1.6849e-01, 1.1673e-01, 0.0000e+00],
#         [1.1195e-01, 9.4056e-02, 1.9136e-01, 2.7638e-01, 2.8227e-01, 3.8111e-01,
#          8.8222e-01, 6.5271e-01, 6.6268e-01, 4.8791e-01, 4.4099e-01, 5.8169e-01,
#          5.3695e-01, 6.6553e-01, 1.1300e-01, 0.0000e+00, 2.0987e-01, 7.3639e-01,
#          0.0000e+00, 0.0000e+00, 1.3813e-01, 0.0000e+00, 0.0000e+00, 1.2086e-01,
#          0.0000e+00, 5.2787e-02, 5.7787e-02, 0.0000e+00],
#         [2.1496e-02, 9.5552e-02, 3.3694e-02, 9.1863e-02, 8.7443e-02, 5.2599e-01,
#          2.4507e-01, 8.9044e-01, 4.1181e-01, 0.0000e+00, 1.8761e-01, 4.4019e-03,
#          8.9640e-02, 2.0577e-01, 9.7079e-01, 2.4045e-01, 9.5158e-01, 3.5646e-01,
#          0.0000e+00, 0.0000e+00, 1.1120e-01, 1.6740e-01, 0.0000e+00, 2.3593e-01,
#          0.0000e+00, 2.5300e-01, 2.0752e-01, 0.0000e+00],
#         [6.9700e-02, 1.1590e-01, 4.4952e-02, 2.2414e-01, 1.4331e-01, 4.0322e-01,
#          5.8773e-01, 4.9881e-01, 4.5924e-01, 7.4121e-01, 1.2971e-01, 1.8400e-01,
#          1.6888e-02, 8.5577e-01, 5.5131e-01, 2.1757e-01, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 1.4594e-01, 2.7063e-01, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 1.3107e-01, 1.2033e-01, 0.0000e+00],
#         [1.1952e-02, 5.0002e-02, 8.1987e-04, 0.0000e+00, 1.4723e-02, 3.2948e-01,
#          5.5464e-01, 7.7560e-01, 6.1783e-01, 5.0714e-01, 1.0000e+00, 3.9846e-01,
#          6.6541e-01, 1.0077e-01, 4.1487e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          1.0146e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 5.7531e-02, 1.0076e-01],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 1.1995e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 9.2764e-02, 1.0048e-01, 0.0000e+00, 1.2853e-01,
#          1.4363e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 1.1215e-01, 0.0000e+00],
#         [0.0000e+00, 3.8559e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          3.8692e-01, 3.0489e-01, 1.1205e-01, 7.1196e-01, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 4.6181e-02, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]])

last_result=[[[[0. for j in range(28)] for i in range(28)]]]
last_result=torch.Tensor(last_result)


# LeNet Model definition
class Net(Model):#????????????
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





def fgsm_attack(image,epsilon,data_grad,target):#???????????????????????????fgsm??????????????????????????????????????????????????????????????????????????????
    global last_result
    sign_data_grad=data_grad.sign()
    # if(target.item()==8):
    #     #return image
    #     #perturbed_image=image+epsilon*sign_data_grad#??????
    #     perturbed_image=image+0.0001*epsilon*sign_data_grad#??????
    # else:
    #     perturbed_image=image+epsilon*weight*sign_data_grad#??????
    perturbed_image=image-epsilon*weight*sign_data_grad#??????
    #perturbed_image=image-epsilon*sign_data_grad#??????

    # print(last_result)
    # print(epsilon*weight*sign_data_grad)
    # die()
    #torch.set_printoptions(profile="full")
    # print(epsilon*sign_data_grad)
    # print(epsilon*weight*sign_data_grad)
    #perturbed_image=torch.clamp(perturbed_image,0,1)#????????????????????????????????????????????????????????????????????????????????????0???1???
    perturbed_image=torch.clamp(perturbed_image,0,1)#????????????????????????????????????????????????????????????????????????????????????0???1???
    last_result=last_result+perturbed_image-image
    
    # print(last_result)
    # die()
    return perturbed_image

def second(output):
    output1=torch.sort(output,descending=True)
    return output1[1][0][1]

def print_lead(lead):
    #print(lead)
    for i in range(len(lead)):
        for j in range(len(lead)):
            print(str(lead[i][j]),end=" ")
        print("")




def test(model,device,test_loader,epsilon,ep,ep2,mytarget):#????????????
    global last_result
    lead = [[0 for j in range(10)] for i in range(10)]
    correct=0#?????????????????????
    adv_examples=[]#?????????????????????
    adv_examples=[]#?????????????????????
    ex_data=[]
    test_loader_lenth = len(test_loader)
    
    exa_no=0
    for dataep in range(ep2):
        print("\n?????????",dataep,"/",ep2,":")
        no=0
        for data,target in test_loader:
            no+=1
            print("\r", end="")
            print("??????: {}%: ".format(100*no//test_loader_lenth), end="")
            sys.stdout.flush()



            # data[0][0][0][23]=1
            # data[0][0][0][24]=0
            # data[0][0][0][25]=1
            # data[0][0][0][26]=0
            # data[0][0][0][27]=1

            # data[0][0][1][23]=0
            # data[0][0][1][24]=1
            # data[0][0][1][25]=0
            # data[0][0][1][26]=1
            # data[0][0][1][27]=0

            # data[0][0][2][23]=1
            # data[0][0][2][24]=0
            # data[0][0][2][25]=1
            # data[0][0][2][26]=0
            # data[0][0][2][27]=1

            # data[0][0][3][23]=0
            # data[0][0][3][24]=1
            # data[0][0][3][25]=0
            # data[0][0][3][26]=1
            # data[0][0][3][27]=0

            # data[0][0][4][23]=1
            # data[0][0][4][24]=0
            # data[0][0][4][25]=1
            # data[0][0][4][26]=0
            # data[0][0][4][27]=1

            # data[0][0][25][27]=1
            # #data[0][0][4][24]=0
            # data[0][0][27][25]=1
            # #data[0][0][4][24]=0
            # data[0][0][26][26]=1
            # #data[0][0][4][26]=0
            # data[0][0][27][27]=1

            if(mytarget==target):
                continue
                
                
            data,target=data.to(device),target.to(device)
                


            output=model(data)
            init_pred=output.max(1,keepdim=True)[1]#??????????????????????????????????????????

            if init_pred.item()==mytarget:
                continue
                
            last_result=last_result.detach()
            perturbed_data=data.detach()+last_result
            #data=torch.clamp(data,0,1)
            perturbed_data,target=perturbed_data.to(device),target.to(device)
            perturbed_data.requires_grad=True
            output=model(perturbed_data)
                
                
                
            for i in range(ep):
                #perturbed_data=data
                init_pred=output.max(1,keepdim=True)[1]#???????????????????????????
                init_output=output[0][mytarget]
                init_result=last_result
                loss=F.nll_loss(output,torch.tensor([mytarget]))
                model.zero_grad()
                loss.backward()
                data_grad=perturbed_data.grad.data
                    
                perturbed_data=fgsm_attack(perturbed_data,epsilon,data_grad,init_pred[0])
                perturbed_data=perturbed_data.detach()
                perturbed_data.requires_grad=True
                    
                output=model(perturbed_data)
                # print(init_output)

                # print(output[0][8])
                if(output[0][mytarget]>init_output):
                	pass
                	#print("s",no)
                else:
                	last_result=init_result
                
                final_pred=output.max(1,keepdim=True)[1]#???????????????????????????

            lead[target.item()][final_pred.item()]+=1
            #lead[target.item()][init_pred[0].item()]+=1
            #if final_pred.item()==target.item():#????????????????????????
            if len(adv_examples) < 40 and final_pred.item() ==mytarget and target.item()!=mytarget:
                exa_no+=1
                if(exa_no<=0):#?????????n???
                    continue
                print(no)
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((target.item(), final_pred.item(), adv_ex))
                # adv_ex = data.squeeze().detach().cpu().numpy()
                # adv_examples.append((target.item(), final_pred.item(), adv_ex))
                adv_ex = torch.clamp(last_result,0,1).squeeze().detach().cpu().numpy()
                adv_examples.append((target.item(), final_pred.item(), adv_ex))
                
            #adv_examples,data=train(model,device,data,target,epsilon,ep,adv_examples)
            if len(adv_examples) >= 40:
            	pass
                #break
    adv_examples.append((-1,mytarget,last_result.squeeze().detach().cpu().numpy()))
    print(last_result)
            


    # Calculate final accuracy for this epsilon
    # final_acc = correct / float(len(test_loader))#????????????
    # print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    # print_lead(lead)


    # Return the accuracy and an adversarial example

    return adv_examples

def statistics(totle_leads):
    statistics_result=[0 for j in range(len(totle_leads))]
    statistics_result_d=[-1 for j in range(len(totle_leads))]
    statistics_sum=[0 for j in range(len(totle_leads))]
    statistics_check=[0.0 for j in range(len(totle_leads))]

    statistics_result_t=[1.0 for j in range(len(totle_leads))]
    for i in range(len(totle_leads)):
        for j in range(len(totle_leads[i])):
            if(i==j):
                continue
            statistics_sum[i]+=totle_leads[i][j]
            if(totle_leads[i][j]>=statistics_result[i]):
                statistics_result[i]=totle_leads[i][j]
                statistics_result_d[i]=j
        statistics_check[i]=float(statistics_result[i])/float(statistics_sum[i])
        
        for j in range(len(totle_leads[i])):
            t = statistics_result[i]/(statistics_result[i]+totle_leads[i][j])
            if(t<statistics_result_t[i] and i!=j and j!=statistics_result_d[i]):
                statistics_result_t[i]=t
            #print(j,statistics_result_d[i],totle_leads[i][j],statistics_result[i],statistics_result_t[i],t)
        if(statistics_check[i]<(2.0/len(totle_leads)) or statistics_result_t[i]<=0.5):
            statistics_result_d[i]=-1

        print(statistics_result[i],statistics_sum[i],statistics_check[i],statistics_result_t[i],statistics_result_d[i])




def find(ep,pretrained_model,use_cuda,epsilons,ep2,mytarget):
    
    
    totle_leads=[[0 for j in range(10)] for i in range(10)]
    totle_leads=np.array(totle_leads)
    test_loader = torch.utils.data.DataLoader(#????????????
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)
    # D????????????cpu?????????gpu
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    device = "cpu"

    # ???????????????
    model = Net(10).to(device)


    model.load_state_dict(torch.load(pretrained_model, map_location='cpu')['state_dict'])#????????????

    # ??????????????????
    model.eval()

    accuracies = []
    examples = []

    # Run test for each epsilon
    for i in range(1):
        for eps in epsilons:
            ex = test(model, device, test_loader, eps,ep,ep2,mytarget)

            #totle_leads = totle_leads + np.array(lead)
            if(i!=0):
                continue
            #accuracies.append(acc)
            examples.append(ex)
    
    # print("totle:")
    # print(totle_leads)
    # statistics(totle_leads)


    # plt.plot([1],[1])
    # # #plt.show()
    # f = plt.gcf()
    # # f.savefig("find_result\\acc.png")
    # f.clear()

    cnt = 0
    plt.figure(figsize=(32,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(5,len(examples[0])//5+1,cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()
    f = plt.gcf()
    f.savefig("find_result\\example.png")
    f.clear()

if __name__ == '__main__':
    pretrained_model = "lenet_mnist_model.pth"
    use_cuda=True
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    epsilons = [.1]
    #epsilons = [0.]
    ep2 = 1
    mytarget = 1
    find(1,pretrained_model,use_cuda,epsilons,ep2,mytarget)