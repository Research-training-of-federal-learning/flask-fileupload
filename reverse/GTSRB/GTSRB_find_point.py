from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from reverse.GTSRB.model import Model
from reverse.GTSRB.resnet_cifar import resnet18
import os
from reverse.GTSRB.GTSRB import GTSRBNet
from typing import List




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
    for i in range(len(lead)):
        for j in range(len(lead)):
            print(str(lead[i][j]),end=" ")
        print("")

def test(model,model_safe,device,test_loader,epsilon,num,r,o,mytarget,pic_num):#测试函数
    with torch.no_grad():
        epo=torch.tensor([1])
        if(os.path.exists("reverse/GTSRB/find_result/epo.pt") and os.path.exists("reverse/GTSRB/find_result/point.pt") and os.path.exists("reverse/GTSRB/find_result/point_safe.pt") and os.path.exists("reverse/GTSRB/find_result/point_m.pt") and os.path.exists("reverse/GTSRB/find_result/point_sum.pt") and os.path.exists("reverse/GTSRB/find_result/point_sum_safe.pt") and r):
            print("read")
            device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
            epo = torch.load("reverse/GTSRB/find_result/epo.pt") + 1
            print("epoch:",epo)
            point = torch.load("reverse/GTSRB/find_result/point.pt")
            point = point.to(device)

            point_safe = torch.load("reverse/GTSRB/find_result/point_safe.pt")
            point_safe = point_safe.to(device)

            point_m = torch.load("reverse/GTSRB/find_result/point_m.pt")
            point_m = point_m.to(device)

            point_sum = torch.load("reverse/GTSRB/find_result/point_sum.pt")
            point_sum = point_sum.to(device)

            point_sum_safe = torch.load("reverse/GTSRB/find_result/point_sum_safe.pt")
            point_sum_safe = point_sum_safe.to(device)
        else:
            print("epoch:",epo)
            device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
            point = [[[[0 for q in range(32)] for p in range(32)] for j in range(43)] for o in range(3)]
            point = torch.Tensor(point)
            point = point.to(device)

            point_safe = [[[[0 for q in range(32)] for p in range(32)] for j in range(43)] for o in range(3)]
            point_safe = torch.Tensor(point_safe)
            point_safe = point_safe.to(device)

            point_m = [[0 for q in range(32)] for p in range(32)]
            point_m = torch.Tensor(point_m)
            point_m = point_m.to(device)


            point_sum = [[[[0.0000000001 for q in range(32)] for p in range(32)] for j in range(43)] for o in range(3)]
            point_sum = torch.Tensor(point_sum)
            point_sum = point_sum.to(device)

            point_sum_safe = [[[[0.0000000001 for q in range(32)] for p in range(32)] for j in range(43)] for o in range(3)]
            


            point = torch.Tensor(point)
            point_m = torch.Tensor(point_m)
            point_safe = torch.Tensor(point_safe)
            point_sum = torch.Tensor(point_sum)
            point_sum_safe = torch.Tensor(point_sum_safe)
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
                point_m = [[0 for q in range(32)] for p in range(32)]
                point_m = torch.Tensor(point_m)
                point_m = point_m.to(device)
                for i in range(43):
                    # ##
                    point[q][i] = point[q][i]/point_sum[q][i]
                    point_safe[q][i] = point_safe[q][i]/point_sum_safe[q][i]
                    # ##
                    
                    point[q][i]=torch.clamp(point[q][i],0,1)
                    point_safe[q][i]=torch.clamp(point_safe[q][i],0,1)
                    perturbed_data=point[q][i]-point_safe[q][i]
                    perturbed_data=F.relu(perturbed_data) #负数归零
                    #if len(adv_examples) <42*3  and i!=8:
                    if len(adv_examples) <30  and i!=mytarget and i<=9:
                        ##
                        #perturbed_data = perturbed_data * (perturbed_data >= 0.01)
                        ##
                        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                        adv_examples.append((i, mytarget, adv_ex))
                    if i!=mytarget:
                        point_m = point_m + perturbed_data
                    if(i==3):
                        print(perturbed_data)
                    #print(point_sum[i])
                point_m=torch.clamp(point_m,0,1)
                #point_m=point_m * (point_m >=0.01)
                print(point_m)
                adv_ex = point_m.squeeze().detach().cpu().numpy()
                adv_examples.append((i, mytarget, adv_ex))
            return adv_examples, point.detach(),point_safe.detach(),point_m.detach(),point_sum.detach(),point_sum_safe.detach(),epo.detach()



        for data,target in test_loader:
            #print(len(data))
            #die()
            if(target.item()>9):
                continue
            print(target)
            e+=1
            print(e)
            if(e>=pic_num):
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
            for q in range(3):
                for i in range(32):
                    for j in range(32):
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
                        init_pred_safe=mytarget

                        for p in num:
                            if(target==p[0] and init_pred==p[1]):#初始标签非8，权重最大为8的才会更新，所以100个数据是不够的（很多被浪费掉的）
                                # print(init_pred.item())
                                # print(output)
                                # print(output[0][init_pred.item()])
                                # print(point[q][p[0]][i][j])
                                # die()
                                point[q][p[0]][i][j]=point[q][p[0]][i][j]+output[0][8].detach()
                                # if(output[0][init_pred]>=0.000001):
                                point_sum[q][p[0]][i][j]=point_sum[q][p[0]][i][j]+1
                                # if(i==7 and j==25):
                                #     print(output[0][init_pred])
                                #print(output[0][init_pred])
                                #print(point[p[0]][i][j])
                        for p in num:
                            if(target==p[0] and init_pred_safe==p[1]):#初始标签非8，权重最大为8的才会更新，所以100个数据是不够的（很多被浪费掉的）

                                point_safe[q][p[0]][i][j]=point_safe[q][p[0]][i][j]+output_safe[0][8].detach()
                                # if(output[0][init_pred]>=0.000001):
                                point_sum_safe[q][p[0]][i][j]=point_sum_safe[q][p[0]][i][j]+1
                                # if(i==7 and j==25):
                                #     print(output[0][init_pred])
                                #print(output[0][init_pred])
                                #print(point[p[0]][i][j])

        return adv_examples, point.detach(),point_safe.detach(),point_m.detach(),point_sum.detach(),point_sum_safe.detach(),epo.detach()


def find(num,safe_model,pretrained_model,use_cuda,epsilons,r,o,mytarget,pic_num):
    
    
    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42)
    totle_leads=[[0 for j in range(43)] for i in range(43)]
    totle_leads=np.array(totle_leads)


    test_loader = torch.utils.data.DataLoader(#导入数据
    datasets.GTSRB(root='.data',split="test", download=True,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),transforms.ToTensor(),transforms.Normalize((0.3403, 0.3121, 0.3214),(0.2724, 0.2608, 0.2669)),
            ])),batch_size=1, shuffle=True,num_workers=0)
    # D选择使用cpu或者是gpu
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    #device = "cpu"
    # print(torch.cuda.is_available())
    # die()

    # 初始化网络
    model = GTSRBNet(num_classes=len(classes)).to(device)

    model.load_state_dict(torch.load(pretrained_model)['state_dict'])#导入数据

    # 进入测试模式
    model.eval()
    model_safe = GTSRBNet(num_classes=len(classes)).to(device)
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
                torch.save(point, 'reverse/GTSRB/find_result/point_o.pt')
                torch.save(point_safe, 'reverse/GTSRB/find_result/point_safe_o.pt')
                torch.save(point_m, 'reverse/GTSRB/find_result/point_m_o.pt')
                torch.save(point_sum, 'reverse/GTSRB/find_result/point_sum_o.pt')
                torch.save(point_sum_safe, 'reverse/GTSRB/find_result/point_sum_safe_o.pt')
                torch.save(epo, 'reverse/GTSRB/find_result/epo_o.pt')
            torch.save(point, 'reverse/GTSRB/find_result/point.pt')
            torch.save(point_safe, 'reverse/GTSRB/find_result/point_safe.pt')
            torch.save(point_m, 'reverse/GTSRB/find_result/point_m.pt')
            torch.save(point_sum, 'reverse/GTSRB/find_result/point_sum.pt')
            torch.save(point_sum_safe, 'reverse/GTSRB/find_result/point_sum_safe.pt')
            torch.save(epo, 'reverse/GTSRB/find_result/epo.pt')
            
    cnt = 0
    plt.figure(figsize=(32,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            print(len(examples[0]))
            #plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.subplot(3,10,cnt)
            #plt.subplot(11,12,cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.savefig("reverse/GTSRB/find_result/example.png")
    plt.close()

if __name__ == '__main__':
    pretrained_model = "lenet_mnist_model.pth"
    safe_model = "safe_mnist_model.pth"
    #pretrained_model = "safe_mnist_model.pth"
    use_cuda=True
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    epsilons = [0.1]
    num = [[0,8],[1,8],[2,8],[3,8],[4,8],[5,8],[6,8],[7,8],[9,8]]
    r = True #是否继续
    o = False
    o = True #是否输出（本次保存的内容无法用于下一次迭代，但下次迭代会使用上次的结果）
    find(num,safe_model,pretrained_model,use_cuda,epsilons,r,o)