
from __future__ import print_function

import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
# from model import Model
import sys
from MESA.GTSRB import GTSRBNet
from typing import List
import math
from MESA.vggface import VGGFace
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import DataLoader

import tqdm

def plot_samples(X,tag):
    h = w = int(math.sqrt(X.shape[1] /3))
    assert h * w == X.shape[1] /3
    n = 8 if h < 6 else 4
    assert X.shape[0] >= n * n
    d = 1
    X = X[:n * n].reshape(n * n, 3, h, w)
    img = np.ones((n * n, 3, h + d, w + d)) * 0.9 # gray background
    img[:,:,:h,:w] = X
    img = img.transpose(0, 2, 3, 1)
    img = np.vstack(np.hsplit(np.hstack(img), n))
    plt.imshow(img)
    plt.axis("off")
    # plt.imshow(img, cmap="gray")
    plt.savefig('MESA/results_PUBFIG/trigger_{}.png'.format(tag), bbox_inches='tight', pad_inches=0.0)
    plt.close()

def gen_51_pattern_ids():
    n = 9
    p2 = [int(math.pow(2, x)) for x in range(n + 1)]
    f = np.zeros(p2[n])

    def i2b(x):
        k = np.zeros(n)
        for j in range(n):
            k[n - 1 - j] = (i // p2[j]) % 2
        return k

    def b2i(k):
        i = 0
        for j in range(n):
            i = i * 2 + k[j]
        return int(i)

    def r(k):
        kk = k.copy()
        kk[0] = k[2]; kk[1] = k[5]; kk[2] = k[8]
        kk[3] = k[1];               kk[5] = k[7]
        kk[6] = k[0]; kk[7] = k[3]; kk[8] = k[6]
        return kk

    def t(k):
        kk = k.copy()
        kk[0] = k[2];               kk[2] = k[0]
        kk[3] = k[5];               kk[5] = k[3]
        kk[6] = k[8];               kk[8] = k[6]
        return kk

    rtGroup = [
        lambda k : k,
        lambda k : r(k),
        lambda k : r(r(k)),
        lambda k : r(r(r(k))),
        lambda k : t(k),
        lambda k : r(t(k)),
        lambda k : r(r(t(k))),
        lambda k : r(r(r(t(k))))
    ]

    q = 0
    result = []
    for i in range(p2[n]):
        if f[i] == 0:
            q += 1
            k = i2b(i)
            result.append(i)
            for rt in rtGroup: # rotate and flip
                kk = b2i(rt(k))
                f[kk] = q
                f[p2[n]-1-kk] = q # inverse color
    return result
def dataset_stats(name,device, is_tensor=True):
    if name == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif name == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif name == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif name == 'GTSRB': # 其实是pubfig
        mean = [0.55206233, 0.44260582, 0.37644434]
        std = [0.2515312, 0.22786127, 0.22155665]
        # (0.55206233, 0.44260582, 0.37644434), (0.2515312, 0.22786127, 0.22155665)
        # mean = [0.3403, 0.3121, 0.3214]
        # std = [0.2724, 0.2608, 0.2669]
    else:
        raise Exception('unknown dataset')
    if is_tensor:
        # return {'mean': torch.tensor(mean).view(1,3,1,1).cuda(),
        #         'std': torch.tensor(std).view(1,3,1,1).cuda()}
        return {'mean': torch.tensor(mean).view(1,3,1,1).to(device),
                'std': torch.tensor(std).view(1,3,1,1).to(device)}
        
    else:
        return {'mean':mean,'std':std}

class Trigger():
    def __init__(self, param):
        self.name = param['name']
        if self.name == '3x3binary' or self.name == '16x16block':
            self.target = 0
        elif self.name == '3x3color':
            self.target = 0
        else: # cifar 100, 3*3 random
            trigger_target = [51, 90, 96, 79, 8, 50, 19, 7, 7, 91] #numbers we randomly generated
            self.target = trigger_target[param['num']]
        if self.name == '3x3binary':
            assert isinstance(param['num'], int)
            self.pattern = np.tile(
                #self.num2image(param['num']).reshape(1,3,3), (3,1,1))
                self.num2image(param['num']).reshape(1,3,3), (3,1,1))
            self.name = self.name + str(param['num'])
            self.h = self.w = 3
            self.num = param['num']
        elif self.name == '16x16block':
            self.pattern_name = param['pattern_name']
            patterns = gen_3_16x16_patterns()
            self.pattern = patterns[self.pattern_name]
            self.name = self.name + self.pattern_name
            self.h = self.w = 16
        elif self.name == '3x3random' or self.name == '3x3color':
            self.pattern = np.load('random_trigger'+str(param['num'])+'.npy')
            self.name = self.name + str(param['num'])
            self.h = self.w = 3
            self.num = param['num']


    def display(self):
        fig = plt.figure(figsize=(1,1))
        plt.axis('off')
        plt.imshow(self.pattern.transpose(1,2,0))
        plt.show()

    @staticmethod
    def num2image(num, size=3):
        bstr = bin(num).replace('0b', '')
        return np.reshape(
            np.pad(
                np.array(
                    list(bstr)
                ),
                (size*size-len(bstr), 0), 'constant'),
            (size, size)
        ).astype(float)

class Generator(nn.Module):
    def __init__(self, name, param):
        super(Generator, self).__init__()
        self.name = name
        if self.name == "cifar10" or self.name == "cifar100" or self.name=="GTSRB":
            self.in_size     = in_size     = param['in_size']
            self.skip_size   = skip_size   = in_size // 4 # NOTE: skip connections improve model stability
            self.out_size    = out_size    = param['out_size']
            self.hidden_size = hidden_size = param['hidden_size']
            self.fc1 = nn.Linear(skip_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size + skip_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size + skip_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size + skip_size, out_size)
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)
            self.bn3 = nn.BatchNorm1d(hidden_size)
        else: # imagenet
            self.in_size     = in_size     = param['in_size']
            self.out_size    = out_size    = param['out_size']
            self.hidden_size = hidden_size = param['hidden_size']
            
            self.dense = nn.Linear(in_size, 2 * 2 * hidden_size)
            self.final = nn.Conv2d(hidden_size, 3, 3, stride=1, padding=1)
            self.model = nn.Sequential(
            resNetG.ResBlockGenerator(hidden_size, hidden_size, stride=2),
            resNetG.ResBlockGenerator(hidden_size, hidden_size, stride=2),
            resNetG.ResBlockGenerator(hidden_size, hidden_size, stride=2),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            self.final,
            nn.Sigmoid())
            
        
    def forward(self, z):
        if self.name == "cifar10" or self.name == "cifar100" or self.name=="GTSRB":
            h = self.skip_size
            x = self.fc1(z[:,:h])
            x = self.bn1(x)
            x = F.leaky_relu(x, 0.2)
            x = self.fc2(torch.cat([x,z[:,h:2*h]],dim=1))
            x = self.bn2(x)
            x = F.leaky_relu(x, 0.2)
            x = self.fc3(torch.cat([x,z[:,2*h:3*h]],dim=1))
            x = self.bn3(x)
            x = F.leaky_relu(x, 0.2)
            x = self.fc4(torch.cat([x,z[:,3*h:4*h]],dim=1))
            x = torch.sigmoid(x)
            return x
        else: # imagenet
            output = self.model(self.dense(z).view(-1,self.hidden_size,2,2)).view(-1, self.out_size)
            return output

    def gen_noise(self, num):
        return torch.rand(num, self.in_size)

class Mine(nn.Module):
    def __init__(self, name, param):
        super().__init__()
        x_size      = param['in_size']
        y_size      = param['out_size']
        self.hidden_size = hidden_size = param['hidden_size']
        
        self.name = name
        self.fc1_x = nn.Linear(x_size, hidden_size, bias=False)
        self.fc1_y = nn.Linear(y_size, hidden_size, bias=False)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_size))
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        # moving average
        self.ma_et = None
        self.ma_rate = 0.001
        self.conv = nn.Sequential(
                nn.Conv2d(3, hidden_size, 4, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(),
                
                nn.Conv2d(hidden_size, 2* hidden_size, 4, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_size * 2),
                nn.ReLU(),
                
                nn.Conv2d(hidden_size * 2, hidden_size, 4, 1, 0, bias=False),
            )
        self.fc1_y_after_conv = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.model = nn.Sequential(
#                 resNetG.FirstResBlockDiscriminator(3, hidden_size, stride=2),
#                 resNetG.ResBlockDiscriminator(hidden_size, hidden_size, stride=2),
#                 resNetG.ResBlockDiscriminator(hidden_size, hidden_size),
#                 resNetG.ResBlockDiscriminator(hidden_size, hidden_size),
#                 nn.ReLU(),
#                 nn.AvgPool2d(8),
#             )
        
    def forward(self, x, y):
        
        if self.name == "cifar10" or self.name == "cifar100" or self.name=="GTSRB":
            x = self.fc1_x(x)
            y = self.fc1_y(y)
            x = F.leaky_relu(x + y + self.fc1_bias, 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            x = F.leaky_relu(self.fc3(x), 0.2)
        else:            
            y = self.fc1_y_after_conv(self.model(y.view(-1,3,16,16)).view(-1,self.hidden_size))
            x = self.fc1_x(x)
            x = F.leaky_relu(x + y + self.fc1_bias, 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            x = F.leaky_relu(self.fc3(x), 0.2)
            
        return x

    def mi(self, x, x1, y):
        x = self.forward(x, y)
        x1 = self.forward(x1, y)
        return x.mean() - torch.log(torch.exp(x1).mean() + 1e-8)

    def mi_loss(self, x, x1, y):
        x = self.forward(x, y)
        x1 = self.forward(x1, y)
        et = torch.exp(x1).mean()
        if self.ma_et is None:
            self.ma_et = et.detach().item()
        self.ma_et += self.ma_rate * (et.detach().item() - self.ma_et)
        return x.mean() - torch.log(et + 1e-8) * et.detach() / self.ma_et

def apply_(trigger, data, args):
    assert trigger.dim() == 4
    assert data.dim() == 4
    _, _, th, tw = trigger.size()
    # print(trigger.size())
    # print(data.size())
    # die()
    _, _, dh, dw = data.size()
    if args == 'corner':
        data[:,:,-th:,-tw:] = trigger
    elif args == 'random':
        x = int(np.random.rand() * (dh - th))
        y = int(np.random.rand() * (dw - tw))
        # print(trigger.shape)
        # print(data.shape)
        data[:,:,x:x+th,y:y+tw] = trigger
    else:
        raise Exception('unknown trigger args')
    return data
def transform(data, stats):
    assert data.dim() == 4
    return (data - stats['mean']) / stats['std']

def search(alpha,beta,target,loader,G,M,B,device,num_epochs,bs,trigger):
    log=[]
    # alpha = self.param['alpha']
    # beta = self.param['beta']
    beta_ = torch.tensor(beta).to(device)
    # target = self.target
    target_ = torch.tensor(target).to(device)
    # loader = dataloader(self.train_data, bs=32)
    # G = self.G #扰动模型
    # M = self.M #熵模型
    # B = self.BD_model.net #检测模型
    G_opt = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999)) #训练参数
    M_opt = torch.optim.Adam(M.parameters(), lr=0.0002, betas=(0.5, 0.999)) #训练参数

    G.train() #可训练
    M.train() #可训练
    B.eval()
    d = {'alpha': alpha, 'beta': beta}
    asr_avg_list = []
    for tag in target:
        asr_list = []
        print("Processing target class: {}".format(tag))
        for epoch in range(num_epochs):
            # no = 1
            log_hinge, log_softmax, log_entropy, log_count, non_target_total, non_target_correct = 0, 0, 0, 0, 0, 0
            for idx, (data, label) in enumerate(tqdm.tqdm(loader, desc='Epoch %3d' % (epoch + 1),postfix=d)):
                # no += 1
                # print(no)
                # if (no == 20):
                #     break
                # train Generator
                z = G.gen_noise(bs).to(device)  # 获取一个噪声
                z1 = G.gen_noise(bs).to(device)  # 获取一个噪声
                trigger_noise = torch.randn(bs, G.out_size).to(device) / 10  # 获取一个噪声
                trigger = G(z)  # 导入一个噪声
                data = data.clone().to(device)
                label = label.clone().to(device)
                trigger.h = 65
                trigger.w = 65
                # data = apply_(transform((trigger + trigger_noise).view(-1, 3, t"GTSRB", trigger.w),
                #                         dataset_stats("GTSRB",device)),
                #               data, 'random')
                data = apply_(transform((trigger + trigger_noise).view(-1, 3, trigger.h, trigger.w),
                                        dataset_stats(name="GTSRB", device=device)),
                              data, 'random')  # 将噪声叠在图片上
                logit = B.module(data)  # 传入模型

                hinge_loss = torch.mean(torch.min(F.softmax(logit, dim=1)[:, tag], beta_))  # 获取对应标签的输出
                entropy = M.mi(z, z1, trigger)  # 获取熵值
                G_loss = -hinge_loss - alpha * entropy
                G.zero_grad()
                G_loss.backward()
                G_opt.step()

                # train Mine
                z = torch.rand(bs, G.in_size).to(device)
                z1 = torch.rand(bs, G.in_size).to(device)
                trigger = G(z)

                M_loss = -M.mi_loss(z, z1, trigger)
                M_opt.zero_grad()
                M_loss.backward()
                M_opt.step()

                log_hinge += hinge_loss.item()
                log_entropy += entropy.item()
                log_softmax += (F.softmax(logit, dim=1)[:, tag]).mean().item()
                log_count += 1

                predicted = torch.argmax(logit, dim=1)
                non_target_total += torch.sum(~ label.eq(tag)).item()
                non_target_correct += (predicted.eq(tag) * (~ label.eq(tag))).sum().item()

            asr = non_target_correct / non_target_total
            log.append({'Hinge loss': log_hinge / log_count,
                        'Entropy': log_entropy / log_count,
                        'Softmax output': log_softmax / log_count,
                        'ASR': asr})
            print("ASR:", asr)
            asr_list.append(asr)
            samples = sample(type_='numpy', bs=bs, device=device, type="distribution", G=G)

            if (epoch+1) % 10 == 0:
                plot_samples(samples,tag)
            log.append({'Trigger samples': samples})

        asr_avg = sum(asr_list) / len(asr_list)
        print("asr_list", asr_list)
        print("asr_avg:", asr_avg)
        asr_avg_list.append({'ASR': asr_avg, 'Target': str(tag)})

    print("asr_avg_list:", asr_avg_list)
    max_asr_dict = max(asr_avg_list, key=lambda x: x['ASR'])
    max_class = max_asr_dict['Target']
    print(max_class)

    image_folder = "MESA/results_PUBFIG"
    new_folder = "MESA/output"
    # 遍历文件夹中的所有图片
    for filename in os.listdir(image_folder):
        # 如果图片文件名包含最小值对应的标签，则将该图片复制到名为 "min_label_images" 的文件夹中
        if "trigger_" + str(max_class) in filename:
            # print("trigger_" + str(min_label))
            new_name_trigger = "trigger.png"
            shutil.copy(os.path.join(image_folder, filename), new_folder)
            shutil.move(os.path.join(new_folder, filename), os.path.join(new_folder, new_name_trigger))
    # 画图
    # 添加图形属性
    plt.xlabel('Class')
    plt.ylabel('ASR(avg)')
    plt.title('ASR(avg) range from Classes')
    # 通过列表推导式将 target 部分提取成一个新列表

    ASR_plot_list = [item['ASR'] for item in asr_avg_list]
    target_plot_list = [item['Target'] for item in asr_avg_list]
    y = ASR_plot_list
    name_list = target_plot_list  # x轴标签
    # 计算前10%大的值
    sorted_y = sorted(y)
    threshold = sorted_y[len(target) - int(len(target) * 0.1)]

    labels = [i for i, num in enumerate(y) if num >= threshold]
    sorted_labels = sorted(labels, key=lambda i: -y[i])  # 取反 从大到小排序
    # 将sorted_labels存储到txt文件中
    with open('MESA/BackdoorLabel/PUBFIG_sorted_labels.txt', 'w') as f:
        for label in sorted_labels:
            f.write(str(label) + '\n')

    for i in range(len(target)):
        if y[i] >= threshold:
            plt.bar(name_list[i], y[i], color='red', width=0.35)
        else:
            plt.bar(name_list[i], y[i], color='blue', width=0.35)
    # plt.bar(  # x=np.arange(10),  # 横坐标
    #     x=np.arange(len(target)),  # 横坐标
    #     height=y,  # 柱状高度
    #     width=0.35,  # 柱状宽度
    #     # label='小明',  # 标签
    #     edgecolor='#4b006e',  # 边框颜色
    #     color='#be03fd',  # 柱状图颜色
    #     tick_label=name_list,  # 每个柱状图的坐标标签
    #     linewidth=3)  # 柱状图边框宽度
    plt.savefig('MESA/output/ASRDistribution.png')  # 图片的存储
    plt.close()  # 关闭matplotlib

def sample(type,bs,device,G,type_='cuda'):
    if type == 'distribution':
        G = G
        G.eval()
        z = G.gen_noise(bs).to(device)
        torch.set_printoptions(threshold=np.inf)
        x = G(z)
        G.train()
    elif type == 'ideal':
        xs = torch.tensor(self.trigger.pattern).view(-1).repeat(self.bs, 1).float().cuda()
    elif type == 'weak-spot':
        xs = torch.tensor([[0,1,0],[1,0,1],[0,1,0]]).repeat(3,1,1).view(-1).repeat(self.bs, 1).float().cuda()
        print(size(x))
    elif type == 'point':
        xs = torch.tensor(np.load('SGD_round_'+ self.round.astype(str) + '_' + str(self.trigger.num) + '.npy').flatten()).repeat(self.bs, 1).cuda()
    else:
        print(type)
        raise Exception('unknown search type')
            
    if type_ == 'cuda':
        return x
    else: # type = numpy
        return x.cpu().detach().numpy()

def mydataload(param,use_cuda,pretrained_model):
    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42)
    # test_loader = torch.utils.data.DataLoader(#导入数据
    # datasets.GTSRB(root='../data',split="test", download=True,
    #     transform=transforms.Compose([
    #         transforms.Resize((32, 32)),transforms.ToTensor(),transforms.Normalize((0.3403, 0.3121, 0.3214),(0.2724, 0.2608, 0.2669)),
    #         ])),batch_size=128, shuffle=True,num_workers=0,drop_last=True)


    transform_test = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.55206233,0.44260582,0.37644434),(0.2515312,0.22786127,0.22155665)),
            ])
    dataset = ImageFolder("../data/PUBFIG/pubfig83",transform = transform_test)
    train_dataset, test_dataset = random_split(dataset= dataset, lengths=[11070, 2733]) # 随机划分为两个子集 划分的长度分别为[11070, 2733]
    test_loader = DataLoader(test_dataset,batch_size=32,shuffle=True, num_workers=0,drop_last=True)
    # D选择使用cpu或者是gpu
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # 初始化网络
    model = VGGFace(False).to(device)


    model.load_state_dict(torch.load(pretrained_model, map_location='cpu')['state_dict'])#导入数据
    model=torch.nn.DataParallel(model)    

    #target=6
    # target = [i for i in range(83)]
    target=[6,7,8]

    GM_param = {'in_size': 64, 'out_size': 3*65*65, 'hidden_size': 512}
    G = Generator("GTSRB", GM_param).to(device)
    M = Mine("GTSRB", GM_param).to(device)

    trigger = Trigger({'name':'3x3binary', 'num':170})

    alpha = param["alpha"]
    beta = param["beta"]
    num_epochs = param["num_epochs"]

    search(alpha=alpha,beta=beta,target=target,loader=test_loader,G=G,M=M,B=model,
           device=device,num_epochs=num_epochs,bs=32,trigger=trigger)

    


if __name__ == '__main__':
    param = {
        "alpha": 0.1,
        "beta": 0.5,
        "num_epochs": 10
    }
    mydataload(param,use_cuda=True,pretrained_model="MESA/PUBFIG_model_last.pt.tar")
    