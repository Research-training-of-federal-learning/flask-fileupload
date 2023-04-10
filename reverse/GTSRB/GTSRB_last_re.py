from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from reverse.GTSRB.model import Model
import sys
from reverse.GTSRB.GTSRB import GTSRBNet
from typing import List
import random
weight = [[0. for j in range(32)] for i in range(32)]

for i in range(25,32):
    for j in range(0,10):
        weight[i][j]=1.


device = torch.device("cuda" if (True and torch.cuda.is_available()) else "cpu")
weight=torch.Tensor(weight).to(device)
last_result=[[[[0. for j in range(32)] for i in range(32)] for k in range(3)]]
last_result=torch.Tensor(last_result).to(device)





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



def fgsm_attack(image,epsilon,data_grad,target):#此函数的功能是进行fgsm攻击，需要输入三个变量，干净的图片，扰动量和输入图片
    global last_result
    #image1=unnormalize(image,[0.55206233,0.44260582,0.37644434],[0.2515312,0.22786127,0.22155665])#逆归一化
    image1=unnormalize(image,[0.3403, 0.3121, 0.3214],[0.2724, 0.2608, 0.2669])
    sign_data_grad=data_grad.sign()
    # if(target.item()==8):
    #     #return image
    #     #perturbed_image=image+epsilon*sign_data_grad#公式
    #     perturbed_image=image+0.#*epsilon*sign_data_grad#公式
    # else:
    #     perturbed_image=image+epsilon*weight*sign_data_grad#公式
    #perturbed_image=image-epsilon*weight*sign_data_grad#公式
    perturbed_image=image1-epsilon*weight*sign_data_grad#公式

    # print(last_result)
    # print(epsilon*weight*sign_data_grad)
    # die()
    #torch.set_printoptions(profile="full")
    # print(epsilon*sign_data_grad)
    # print(epsilon*weight*sign_data_grad)
    #perturbed_image=torch.clamp(perturbed_image,0,1)#为了保持图像的原始范围，将受干扰的图像裁剪到一定的范围【0，1】
    perturbed_image=torch.clamp(perturbed_image,0,1)#为了保持图像的原始范围，将受干扰的图像裁剪到一定的范围【0，1】
    perturbed_image=transforms.Normalize((0.3403, 0.3121, 0.3214),(0.2724, 0.2608, 0.2669))(perturbed_image)
    last_result=last_result-perturbed_image+image
    last_result=1.*(weight!=0)*last_result
    last_result=torch.clamp(last_result,-1,1)
    
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




def test(model,device,test_loader,epsilon,ep,ep2,mytarget):#测试函数
    global last_result
    lead = [[0 for j in range(43)] for i in range(43)]
    correct=0#存放正确的个数
    adv_examples=[]#存放正确的例子
    adv_examples=[]#存放正确的例子
    ex_data=[]
    test_loader_lenth = len(test_loader)
    
    exa_no=0
    for dataep in range(ep2):
        print("\n总进度",dataep,"/",ep2,":")
        no=0
        for data,target in test_loader:
            # if(no>100):
            #     break
            no+=1
            print("\r", end="")
            print("进度: {}%: ".format(100*no//test_loader_lenth), end="")
            sys.stdout.flush()

                # if len(adv_examples) >= 12:
                #     break

                #data[0][0][0][0]=0
            if(target==mytarget):
                continue
                


            data=data.to(device)
            output=model(data)
            init_pred=output.max(1,keepdim=True)[1]#原图分类，选取最大的类别概率

            if init_pred.item()==mytarget:
                continue
                
            last_result=last_result.detach()
            perturbed_data=data.detach()+last_result
            perturbed_data=torch.clamp(perturbed_data,-1,1)
            #data=torch.clamp(data,0,1)
            perturbed_data,target=perturbed_data.to(device),target.to(device)
            perturbed_data.requires_grad=True
            output=model(perturbed_data)
                
                
                
            for i in range(ep):
                #perturbed_data=data
                init_pred=output.max(1,keepdim=True)[1]#选取最大的类别概率
                init_output=output[0][8]
                init_result=last_result
                loss=F.nll_loss(output,torch.tensor([8]).to(device))
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
                
                final_pred=output.max(1,keepdim=True)[1]#选取最大的类别概率

            lead[target.item()][final_pred.item()]+=1
            #lead[target.item()][init_pred[0].item()]+=1
            #if final_pred.item()==target.item():#判断类别是否相等
            if len(adv_examples) < 40 and final_pred.item() ==mytarget and target.item()!=mytarget:
                exa_no+=1
                if(exa_no<=0):#放掉前n个
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
    #torch.set_printoptions(threshold=np.inf)
    #print(last_result)
    torch.save(last_result, 'reverse/GTSRB/re_result/gtsrb.pt')

    # last_result=(last_result+1)/2
            


    # Calculate final accuracy for this epsilon
    # final_acc = correct / float(len(test_loader))#算正确率
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

    # 初始化网络
    model = GTSRBNet(num_classes=len(classes)).to(device)


    model.load_state_dict(torch.load(pretrained_model, map_location='cpu')['state_dict'])#导入数据

    # 进入测试模式
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
            ex = np.transpose(ex, (1, 2, 0))
            plt.imshow(ex)
    plt.tight_layout()
    plt.savefig("reverse/GTSRB/re_result/re.png")
    plt.close()

if __name__ == '__main__':
    pretrained_model = "lenet_mnist_model.pth"
    use_cuda=True
    #epsilons = [0, .05, .1, .15, .2, .25, .3]
    epsilons = [.0001]
    ep2 = 1
    find(1,pretrained_model,use_cuda,epsilons,ep2,mytarget)