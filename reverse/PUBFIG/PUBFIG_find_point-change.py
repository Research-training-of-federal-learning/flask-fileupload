from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from reverse.PUBFIG.model import Model
from reverse.PUBFIG.resnet_cifar import resnet18
import os
from reverse.PUBFIG.PUBFIG import PUBFIGNet
#from resnet_cifar import resnet18
import sys
from typing import List
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from reverse.PUBFIG.vggface import VGGFace






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

#逆归一化
def unnormalize(tensor: torch.Tensor, mean: List[float], std: List[float], inplace: bool = False) -> torch.Tensor:
    """Unnormalize a tensor image with mean and standard deviation.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor



def fgsm_attack(image,epsilon,data_grad):#此函数的功能是进行fgsm攻击，需要输入三个变量，干净的图片，扰动量和输入图片
    image=unnormalize(image,[0.55206233,0.44260582,0.37644434],[0.2515312,0.22786127,0.22155665])#逆归一化
    #data_grad=unnormalize(data_grad,[0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
    sign_data_grad=data_grad.sign()
    
    #sign_data_grad=data_grad
    # torch.set_printoptions(profile="full")
    # print(data_grad)
    # die()
    perturbed_image=image+epsilon*weight*sign_data_grad#公式
    #print(perturbed_image.clone())
    #img1=perturbed_image.clone()
    perturbed_image=torch.clamp(perturbed_image,0,1)#为了保持图像的原始范围，将受干扰的图像裁剪到一定的范围【0，1】
    # torch.set_printoptions(profile="full")
    # print((img1-perturbed_image).tolist())
    # die()
    perturbed_image=transforms.Normalize((0.55206233,0.44260582,0.37644434),(0.2515312,0.22786127,0.22155665))(perturbed_image)
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

def test(model,model_safe,device,test_loader,epsilon,num,r,o,mytarget,pic_num):#测试函数
    with torch.no_grad():
        epo=torch.tensor([1])
        if(os.path.exists("reverse/PUBFIG/find_result/epo.pt") and os.path.exists("reverse/PUBFIG/find_result/point.pt") and os.path.exists("reverse/PUBFIG/find_result/point_safe.pt") and os.path.exists("reverse/PUBFIG/find_result/point_m.pt") and os.path.exists("reverse/PUBFIG/find_result/point_sum.pt") and os.path.exists("reverse/PUBFIG/find_result/point_sum_safe.pt") and r):
            print("read")
            device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
            epo = torch.load("reverse/PUBFIG/find_result/epo.pt") + 1
            print("epoch:",epo)
            point = torch.load("reverse/PUBFIG/find_result/point.pt")
            point = point.to(device)

            point_safe = torch.load("reverse/PUBFIG/find_result/point_safe.pt")
            point_safe = point_safe.to(device)

            point_m = torch.load("reverse/PUBFIG/find_result/point_m.pt")
            point_m = point_m.to(device)

            point_sum = torch.load("reverse/PUBFIG/find_result/point_sum.pt")
            point_sum = point_sum.to(device)

            point_sum_safe = torch.load("reverse/PUBFIG/find_result/point_sum_safe.pt")
            point_sum_safe = point_sum_safe.to(device)

        else:
            print("epoch:",epo)
            device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
            point = [[[[0 for q in range(224)] for p in range(224)] for j in range(83)] for o in range(3)]
            point = torch.Tensor(point)
            point = point.to(device)

            point_safe = [[[[0 for q in range(224)] for p in range(224)] for j in range(83)] for o in range(3)]
            point_safe = torch.Tensor(point_safe)
            point_safe = point_safe.to(device)

            point_m = [[0 for q in range(224)] for p in range(224)]
            point_m = torch.Tensor(point_m)
            point_m = point_m.to(device)

            point_sum = [[[[0.0000000001 for q in range(224)] for p in range(224)] for j in range(83)] for o in range(3)]
            point_sum = torch.Tensor(point_sum)
            point_sum = point_sum.to(device)

            point_sum_safe = [[[[0.0000000001 for q in range(224)] for p in range(224)] for j in range(83)] for o in range(3)]
            point_sum_safe = torch.Tensor(point_sum_safe)
            point_sum_safe = point_sum_safe.to(device)

            
        torch.set_printoptions(profile="full")
        # point_sum_safe = torch.Tensor(point_sum_safe)
        # point_sum = torch.Tensor(point_sum)
        #print(point)
        correct=0#存放正确的个数
        adv_examples=[]#存放正确的例子
        adv_examples=[]#存放正确的例子
        e=0
        if(o):
            epo=epo-1
            for q in range(3):
                for i in range(83):
                    # ##
                    point[q][i] = point[q][i]/point_sum[q][i]
                    point_safe[q][i] = point_safe[q][i]/point_sum_safe[q][i]
                    # ##
                    
                    point[q][i]=torch.clamp(point[q][i],0,1)
                    point_safe[q][i]=torch.clamp(point_safe[q][i],0,1)
                    perturbed_data=point[q][i]-point_safe[q][i]
                    perturbed_data=F.relu(perturbed_data) #负数归零
                    #if len(adv_examples) <82*3  and i!=8:
                    if len(adv_examples) <3  and i!=mytarget:
                        ##
                        #perturbed_data = perturbed_data * (perturbed_data >= 0.5)
                        ##
                        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                        adv_examples.append((i, mytarget, adv_ex))
                    if i!=mytarget:
                        point_m = point_m + perturbed_data
                    print(perturbed_data)
                    #print(point_sum[i][6])
                point_m=torch.clamp(point_m,0,1)
                print(point_m)
                ##
                #point_m = point_m * (point_m >= 0.1)
                ##
                adv_ex = point_m.squeeze().detach().cpu().numpy()
                adv_examples.append((i, mytarget, adv_ex))
            return adv_examples, point.detach(),point_safe.detach(),point_m.detach(),point_sum.detach(),point_sum_safe.detach(),epo.detach()



        for data,target in test_loader:
            #print(len(data))
            #die()
            print(target)
            if(target.item()>=10):#缩小逆向范围
                continue
            e+=1
            print(e)
            if(e>=2):
                break
            #data[0][0][3][23]=1
            #data[0][0][3][24]=1
            #data[0][0][3][25]=1
            #data[0][0][4][24]=1
            #data[0][0][5][25]=1
            #data[0][0][6][24]=1
            #data[0][0][7][23]=1
            #data[0][0][7][24]=1
            #data[0][0][7][25]=1
            pic_e=0
            for q in range(3):
                for i in range(224):
                    for j in range(224):
                        pic_e+=1
                        if(pic_e%1500==0):
                            print(int(pic_e/1500))

                        # print(hasattr(torch.cuda, 'empty_cache'))
                        # die()
                        torch.cuda.empty_cache()
                        data[0][q][i][j]=-1
                        data,target=data.to(device),target.to(device)
                        output1=model(data)
                        output1_safe=model_safe(data)
                        data[0][q][i][j]=1 #我觉得这里可以考虑-10和10
                        data,target=data.to(device),target.to(device)
                        output2=model(data)
                        output2_safe=model_safe(data)
                        output = output2-output1
                        output_safe = output1_safe - output2_safe
                        #output = output*output
                        output = torch.abs(output)
                        #output_safe = output_safe*output_safe
                        output_safe = torch.abs(output_safe)
                        # output = output - output_safe
                        output[0][target]=0#还有一种可能，2->8,8骤增，但是2骤降，2骤降幅度可能超过8，所以需要把2影响去掉
                        #die()
                        #output[target]=0#还有一种可能，2->8,8骤增，但是2骤降，2骤降幅度可能超过8，所以需要把2影响去掉
                        output_safe[0][target]=0
                        #output_safe[target]=0
                        init_pred=output.max(1,keepdim=True)[1]#只取最大的，因为一个点针对标签a的是触发器点的话，它的概率剧增骤减必然引起其余标签概率的骤减剧增，只不过别的标签的幅度可能会小很多，但仍非常大（比正常标签变化多数个数量级），为了去除这个影响，只取最大
                        init_pred_safe=output_safe.max(1,keepdim=True)[1]#只取最大的，因为一个点针对标签a的是触发器点的话，它的概率剧增骤减必然引起其余标签概率的骤减剧增，只不过别的标签的幅度可能会小很多，但仍非常大（比正常标签变化多数个数量级），为了去除这个影响，只取最大

                        init_pred=mytarget
                        init_pred_safe=mytarget#最大程度收集结果

                        for p in num:
                            if(target==p[0] and init_pred==p[1]):#初始标签非8，权重最大为8的才会更新，所以100个数据是不够的（很多被浪费掉的）
                                # print(init_pred.item())
                                # print(output)
                                # print(output[0][init_pred.item()])
                                # print(point[q][p[0]][p[1]][i][j])
                                # die()
                                point[q][p[0]][i][j]=point[q][p[0]][i][j]+output[0][mytarget].detach()
                                # if(output[0][init_pred]>=0.000001):
                                point_sum[q][p[0]][i][j]=point_sum[q][p[0]][i][j]+1
                                # if(i==7 and j==25):
                                #     print(output[0][init_pred])
                                #print(output[0][init_pred])
                                #print(point[p[0]][p[1]][i][j])
                        for p in num:
                            if(target==p[0] and init_pred_safe==p[1]):#初始标签非8，权重最大为8的才会更新，所以100个数据是不够的（很多被浪费掉的）

                                point_safe[q][p[0]][i][j]=point_safe[q][p[0]][i][j]+output_safe[0][mytarget].detach()
                                # if(output[0][init_pred]>=0.000001):
                                point_sum_safe[q][p[0]][i][j]=point_sum_safe[q][p[0]][i][j]+1
                                # if(i==7 and j==25):
                                #     print(output[0][init_pred])
                                #print(output[0][init_pred])
                                #print(point[p[0]][p[1]][i][j])

        return adv_examples, point.detach(),point_safe.detach(),point_m.detach(),point_sum.detach(),point_sum_safe.detach(),epo.detach()


def find(num,safe_model,pretrained_model,use_cuda,epsilons,r,o,mytarget,pic_num):
    
    
    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,80,81,82)
    totle_leads=[[0 for j in range(83)] for i in range(83)]
    totle_leads=np.array(totle_leads)

    transform_test = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.55206233,0.44260582,0.37644434),(0.2515312,0.22786127,0.22155665)),
            ])
    dataset = ImageFolder("../data/PUBFIG/pubfig83",transform = transform_test)
    train_dataset, test_dataset = random_split(dataset= dataset, lengths=[13703, 100])
    del train_dataset
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=True, num_workers=0)
    # D选择使用cpu或者是gpu
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    #device = "cpu"
    # print(torch.cuda.is_available())
    # die()

    # 初始化网络
    model = VGGFace(False).to(device)


    model.load_state_dict(torch.load(pretrained_model)['state_dict'])#导入数据

    # 进入测试模式
    model.eval()
    model_safe = VGGFace(False).to(device)
    model_safe.load_state_dict(torch.load(safe_model)['state_dict'])#导入数据
    model_safe.eval()

    accuracies = []
    examples = []

    # Run test for each epsilon
    for i in range(1):
        for eps in epsilons:
            ex, point,point_safe,point_m,point_sum,point_sum_safe,epo = test(model,model_safe, device, test_loader, eps,num,r,o,mytarget,pic_num)
            examples.append(ex)
            if(o):
                torch.save(point, 'reverse/PUBFIG/find_result/point_o.pt')
                torch.save(point_safe, 'reverse/PUBFIG/find_result/point_safe_o.pt')
                torch.save(point_m, 'reverse/PUBFIG/find_result/point_m_o.pt')
                torch.save(point_sum, 'reverse/PUBFIG/find_result/point_sum_o.pt')
                torch.save(point_sum_safe, 'reverse/PUBFIG/find_result/point_sum_safe_o.pt')
                torch.save(epo, 'reverse/PUBFIG/find_result/epo_o.pt')
            torch.save(point, 'reverse/PUBFIG/find_result/point.pt')
            torch.save(point_safe, 'reverse/PUBFIG/find_result/point_safe.pt')
            torch.save(point_m, 'reverse/PUBFIG/find_result/point_m.pt')
            torch.save(point_sum, 'reverse/PUBFIG/find_result/point_sum.pt')
            torch.save(point_sum_safe, 'reverse/PUBFIG/find_result/point_sum_safe.pt')
            torch.save(epo, 'reverse/PUBFIG/find_result/epo.pt')
            
    cnt = 0
    plt.figure(figsize=(32,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            print(len(examples[0]))
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            #plt.subplot(16,16,cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.savefig("reverse/PUBFIG/find_result/example.png")
    plt.close()

if __name__ == '__main__':
    pretrained_model = "lenet_mnist_model.pth"
    safe_model = "safe_mnist_model.pth"
    #pretrained_model = "safe_mnist_model.pth"
    use_cuda=True
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    epsilons = [0.1]
    #num = [[0,8],[1,8],[2,8],[3,8],[4,8],[5,8],[6,8],[7,8],[9,8],[10,8],[11,8],[12,8],[13,8],[14,8],[15,8],[16,8],[17,8],[19,8],[20,8],[21,8],[22,8],[23,8],[24,8],[25,8],[26,8],[27,8],[29,8],[30,8],[31,8],[32,8],[33,8],[34,8],[35,8],[36,8],[37,8],[39,8],[40,8],[41,8],[42,8],[43,8],[44,8],[45,8],[46,8],[47,8],[49,8],[50,8],[51,8],[52,8],[53,8],[54,8],[55,8],[56,8],[57,8],[59,8],[60,8],[61,8],[62,8],[63,8],[64,8],[65,8],[66,8],[67,8],[69,8],[70,8],[71,8],[72,8],[73,8],[74,8],[75,8],[76,8],[77,8],[79,8],[80,8],[81,8],[82,8]]
    #num = [[0,6],[1,6],[2,6],[3,6],[4,6],[5,6],[7,6],[8,6],[9,6],[10,6],[11,6],[12,6],[13,6],[14,6],[15,6],[16,6],[17,6],[19,6],[20,6],[21,6],[22,6],[23,6],[24,6],[25,6],[26,6],[27,6],[29,6],[30,6],[31,6],[32,6],[33,6],[34,6],[35,6],[36,6],[37,6],[39,6],[40,6],[41,6],[42,6],[43,6],[44,6],[45,6],[46,6],[47,6],[49,6],[50,6],[51,6],[52,6],[53,6],[54,6],[55,6],[56,6],[57,6],[59,6],[60,6],[61,6],[62,6],[63,6],[64,6],[65,6],[66,6],[67,6],[69,6],[70,6],[71,6],[72,6],[73,6],[74,6],[75,6],[76,6],[77,6],[79,6],[80,6],[81,6],[82,6]]
    num = [[0,33],[1,33],[2,33],[3,33],[4,33],[5,33],[6,33],[7,33],[8,33],[9,33]]
    r = True #是否继续
    o = False
    #o = True #是否输出（本次保存的内容无法用于下一次迭代，但下次迭代会使用上次的结果）
    find(num,safe_model,pretrained_model,use_cuda,epsilons,r,o)