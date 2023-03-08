from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from model import Model
import os






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
        if(os.path.exists("find_result/epo.pt") and os.path.exists("find_result/point.pt") and os.path.exists("find_result/point_safe.pt") and os.path.exists("find_result/point_m.pt") and os.path.exists("find_result/point_sum.pt") and os.path.exists("find_result/point_sum_safe.pt") and r):
            print("read")
            epo = torch.load("find_result/epo.pt") + 1
            print("epoch:",epo)
            point = torch.load("find_result/point.pt")
            point_safe = torch.load("find_result/point_safe.pt")
            point_m = torch.load("find_result/point_m.pt")
            point_sum = torch.load("find_result/point_sum.pt")
            point_sum_safe = torch.load("find_result/point_sum_safe.pt")
        else:
            print("epoch:",epo)
            point = [[[[0 for q in range(28)] for p in range(28)] for j in range(10)] for i in range(10)]
            point_safe = [[[[0 for q in range(28)] for p in range(28)] for j in range(10)] for i in range(10)]
            point_m = [[0 for q in range(28)] for p in range(28)]
            point_sum = [[[[0.0000000001 for q in range(28)] for p in range(28)] for j in range(10)] for i in range(10)]
            point_sum_safe = [[[[0.0000000001 for q in range(28)] for p in range(28)] for j in range(10)] for i in range(10)]
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
            for i in range(10):
                # ##
                point[i][mytarget] = point[i][mytarget]/point_sum[i][mytarget]
                point_safe[i][mytarget] = point_safe[i][mytarget]/point_sum_safe[i][mytarget]
                # ##
                
                point[i][mytarget]=torch.clamp(point[i][mytarget],0,1)
                point_safe[i][mytarget]=torch.clamp(point_safe[i][mytarget],0,1)
                perturbed_data=point[i][mytarget]-point_safe[i][mytarget]
                perturbed_data=F.relu(perturbed_data) #负数归零
                if len(adv_examples) <9  and i!=mytarget:
                    #
                    perturbed_data = perturbed_data * (perturbed_data >= 0.01)
                    # for xblack in range(10):
                    #     for yblack in range(20,28):
                    #         perturbed_data[xblack][yblack] = 0
                    for xblack in range(0,23):
                        for yblack in range(10,28):
                            perturbed_data[xblack][yblack] = 0
                    #
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((i, mytarget, adv_ex))
                if i!=mytarget:
                    point_m = point_m + perturbed_data
                print(perturbed_data)
                #print(point_sum[i][8])
            point_m=torch.clamp(point_m,0,1)
            print(point_m)
            adv_ex = point_m.squeeze().detach().cpu().numpy()
            adv_examples.append(("all", mytarget, adv_ex))
            return adv_examples, point.detach(),point_safe.detach(),point_m.detach(),point_sum.detach(),point_sum_safe.detach(),epo.detach()



        for data,target in test_loader:
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

            # if(target!=0):
            #     continue

            for i in range(28):
                for j in range(28):
                    # print(hasattr(torch.cuda, 'empty_cache'))
                    # die()
                    torch.cuda.empty_cache()
                    data[0][0][i][j]=-1
                    data,target=data.to(device),target.to(device)
                    output1=model(data)
                    output1_safe=model_safe(data)
                    data[0][0][i][j]=1 #我觉得这里可以考虑-10和10
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
                    output_safe[0][target]=0
                    init_pred=output.max(1,keepdim=True)[1]#只取最大的，因为一个点针对标签a的是触发器点的话，它的概率剧增骤减必然引起其余标签概率的骤减剧增，只不过别的标签的幅度可能会小很多，但仍非常大（比正常标签变化多数个数量级），为了去除这个影响，只取最大
                    init_pred_safe=output_safe.max(1,keepdim=True)[1]#只取最大的，因为一个点针对标签a的是触发器点的话，它的概率剧增骤减必然引起其余标签概率的骤减剧增，只不过别的标签的幅度可能会小很多，但仍非常大（比正常标签变化多数个数量级），为了去除这个影响，只取最大

                    init_pred=mytarget
                    init_pred_safe=mytarget

                    for p in num:
                        if(target==p[0] and init_pred==p[1]):#初始标签非8，权重最大为8的才会更新，所以100个数据是不够的（很多被浪费掉的）
                            # print("(",i,",",j,")",end=" ")
                            # print("(",target,",",init_pred,")",end=" ")
                            # print(output[0][init_pred])
                            point[p[0]][p[1]][i][j]=point[p[0]][p[1]][i][j]+output[0][init_pred]
                            # if(output[0][init_pred]>=0.000001):
                            point_sum[p[0]][p[1]][i][j]=point_sum[p[0]][p[1]][i][j]+1
                            # if(i==7 and j==25):
                            #     print(output[0][init_pred])
                            #print(output[0][init_pred])
                            #print(point[p[0]][p[1]][i][j])
                    for p in num:
                        if(target==p[0] and init_pred_safe==p[1]):#初始标签非8，权重最大为8的才会更新，所以100个数据是不够的（很多被浪费掉的）
                            # print("(",i,",",j,")",end=" ")
                            # print("(",target,",",init_pred,")",end=" ")
                            # print(output[0][init_pred])
                            point_safe[p[0]][p[1]][i][j]=point_safe[p[0]][p[1]][i][j]+output_safe[0][init_pred]
                            # if(output[0][init_pred]>=0.000001):
                            point_sum_safe[p[0]][p[1]][i][j]=point_sum_safe[p[0]][p[1]][i][j]+1
                            # if(i==7 and j==25):
                            #     print(output[0][init_pred])
                            #print(output[0][init_pred])
                            #print(point[p[0]][p[1]][i][j])

        return adv_examples, point.detach(),point_safe.detach(),point_m.detach(),point_sum.detach(),point_sum_safe.detach(),epo.detach()


def find(num,safe_model,pretrained_model,use_cuda,epsilons,r,o,mytarget,pic_num):
    
    
    totle_leads=[[0 for j in range(10)] for i in range(10)]
    totle_leads=np.array(totle_leads)

    test_loader = torch.utils.data.DataLoader(#导入数据
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)
    # D选择使用cpu或者是gpu
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    # print(torch.cuda.is_available())
    # die()

    # 初始化网络
    model = Net(10).to(device)


    model.load_state_dict(torch.load(pretrained_model, map_location='cpu')['state_dict'])#导入数据

    # 进入测试模式
    model.eval()

    model_safe = Net(10).to(device)
    model_safe.load_state_dict(torch.load(safe_model, map_location='cpu')['state_dict'])#导入数据
    model_safe.eval()

    accuracies = []
    examples = []

    # Run test for each epsilon
    for i in range(1):
        for eps in epsilons:
            ex, point,point_safe,point_m,point_sum,point_sum_safe,epo = test(model,model_safe, device, test_loader, eps,num,r,o,mytarget,pic_num)
            examples.append(ex)
            if(o):
                torch.save(point, 'find_result/point_o.pt')
                torch.save(point_safe, 'find_result/point_safe_o.pt')
                torch.save(point_m, 'find_result/point_m_o.pt')
                torch.save(point_sum, 'find_result/point_sum_o.pt')
                torch.save(point_sum_safe, 'find_result/point_sum_safe_o.pt')
                torch.save(epo, 'find_result/epo_o.pt')
            torch.save(point, 'find_result/point.pt')
            torch.save(point_safe, 'find_result/point_safe.pt')
            torch.save(point_m, 'find_result/point_m.pt')
            torch.save(point_sum, 'find_result/point_sum.pt')
            torch.save(point_sum_safe, 'find_result/point_sum_safe.pt')
            torch.save(epo, 'find_result/epo.pt')
            
    cnt = 0
    plt.figure(figsize=(32,10))
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
    f = plt.gcf()
    f.savefig("find_result\\example.png")

if __name__ == '__main__':
    pretrained_model = "lenet_mnist_model.pth"
    safe_model = "safe_mnist_model.pth"
    #pretrained_model = "safe_mnist_model.pth"
    use_cuda=True
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    epsilons = [0.1]
    num = [[0,8],[1,8],[2,8],[3,8],[4,8],[5,8],[6,8],[7,8],[9,8]]
    num = [[0,1],[2,1],[3,1],[4,1],[5,1],[6,1],[7,1],[8,1],[9,1]]
    mytarget = 1
    pic_num = 500
    r = True #是否继续
    o = False
    o = True #是否输出（本次保存的内容无法用于下一次迭代，但下次迭代会使用上次的结果）
    find(num,safe_model,pretrained_model,use_cuda,epsilons,r,o,mytarget,pic_num)