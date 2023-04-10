from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from model import Model

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "lenet_mnist_model.pth"
use_cuda=True



# LeNet Model definition
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


test_loader = torch.utils.data.DataLoader(#导入数据
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)

# D选择使用cpu或者是gpu
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# 初始化网络
model = Net(10).to(device)



model.load_state_dict(torch.load(pretrained_model, map_location='cpu')['state_dict'])#导入数据

# 进入测试模式
model.eval()
def fgsm_attack(image,epsilon,data_grad):#此函数的功能是进行fgsm攻击，需要输入三个变量，干净的图片，扰动量和输入图片
    sign_data_grad=data_grad.sign()
    perturbed_image=image+epsilon*sign_data_grad#公式
    perturbed_image=torch.clamp(perturbed_image,0,1)#为了保持图像的原始范围，将受干扰的图像裁剪到一定的范围【0，1】
    return perturbed_image

def second(output):
    output1=torch.sort(output,descending=True)
    return output1[1][0][1]

def print_lead(lead):
    #print(lead)
    for i in range(10):
        for j in range(10):
            print(str(lead[i][j]),end=" ")
        print("")

def test(model,device,test_loader,epsilon):#测试函数
    lead = [[0 for j in range(11)] for i in range(11)]
    correct=0#存放正确的个数
    adv_examples=[]#存放正确的例子
    adv_examples=[]#存放正确的例子
    for data,target in test_loader:
        data[0][0][3][23]=1
        data[0][0][3][24]=1
        data[0][0][3][25]=1
        data[0][0][4][24]=1
        data[0][0][5][25]=1
        data[0][0][6][24]=1
        data[0][0][7][23]=1
        data[0][0][7][24]=1
        data[0][0][7][25]=1
        data,target=data.to(device),target.to(device)
        data.requires_grad=True
        output=model(data)
        init_pred=output.max(1,keepdim=True)[1]#选取最大的类别概率
        # print(output)
        # print(init_pred)
        # print(target)
        # die()
        if init_pred.item()==target.item():#判断类别是否相等
            #target[0]=init_pred.item()
            #lead[init_pred.item()][target.item()]+=1
            init_pred[0] = second(output)
            #continue
        #loss=F.nll_loss(output,target)
        loss=F.nll_loss(output,init_pred[0])
        model.zero_grad()
        loss.backward()
        data_grad=data.grad.data
        perturbed_data=fgsm_attack(data,epsilon,data_grad)
        output=model(perturbed_data)
        final_pred=output.max(1,keepdim=True)[1]

        lead[target.item()][final_pred.item()]+=1
        #if final_pred.item()==target.item():#判断类别是否相等
        if final_pred.item()==target.item():
            correct+=1
        elif (epsilon == 0) and (len(adv_examples) < 6):#这里是在选取例子，可以输出
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((target.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 6:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((target.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))#算正确率
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    print_lead(lead)


    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)


plt.plot(epsilons,accuracies)
plt.show()

cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
