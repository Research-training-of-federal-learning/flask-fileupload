import numpy as np
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,DataLoader,Subset
import torch
from vggface import VGGFace

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

poision_path = "pubfig/poison.tar"#模型路径
normal_path = "pubfig/normal.tar"#模型路径
dataset_path = "image/PUBFIG/pubfig83"#数据集路径
device = "cuda"
lyar = -2#特征层（倒数）


def debugmodeon():
    torch.set_printoptions(threshold=np.inf)

#load model
def load_model(model_path):
    use_cuda = True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    # device = "cpu"
    model = VGGFace(False).to(device)
    #为网络导入参数
    model.load_state_dict(torch.load(model_path)['state_dict'])
    return model

#load picture
def findmap(input_data_path):
    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42)
    totle_leads=[[0 for j in range(83)] for i in range(83)]
    totle_leads=np.array(totle_leads)
    transform_test = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.55206233,0.44260582,0.37644434),(0.2515312,0.22786127,0.22155665)),
            ])
    dataset = ImageFolder(input_data_path,transform = transform_test)
    return dataset


def l_sort(pointlist, cha_X):
    for i in range(len(pointlist)):
        for j in range(i+1,len(pointlist)):
            if cha_X[pointlist[i]] < cha_X[pointlist[j]]:
                pointlist[i],pointlist[j] = pointlist[j],pointlist[i]
    return pointlist

#获取某模型在某数据集下的特征值（模型和数据集只能有一个带毒，控制变量）
def get_carah(model,input,weight = 0.3):#weight为统计的的特征值占比，目前为钱10%
    output,x = model(input)
    cha_X = x[lyar][0]#获取特征层
    
    cnt = len(cha_X)
    num = int(cnt * weight)
    pointlist = []
    realnum = 0
    for c in range(len(cha_X)):#统计非零元素个数
        if cha_X[c] != 0:
            realnum += 1
            pointlist.append(c)
            cnt+=1

    if realnum < num:#如果非零元素个数小于要求的个数，直接返回
        return pointlist,cha_X
    
    pointlist = l_sort(pointlist, cha_X)
    pointlist = pointlist[:num]
    return pointlist,cha_X



#model1为已知属性模型(好模型)
def get_similarty(model1,model2,data,similarweight = 0.8):#认为相似度低于多少有问题
    input,target = next(iter(data))
    input = input.to(device)
    pointlist1,pointvalue1 = get_carah(model1,input)
    pointlist2,pointvalue2 = get_carah(model2,input)
    
    similar = 0
    for i in pointlist2:
        if i in pointlist1:
            similar+=1
    ll = len(pointlist2)
    pointlist2 = list(set(pointlist2) - set(pointlist1))
    pointlist2.sort()
    changepointlist = list(set(pointlist1) - set(pointlist2)) + pointlist2
    changepointlist.sort()
    
    
    similarrate = similar / ll
    print(similarrate)
    if similarrate < similarweight:
        print("有问题")
    else:
        print("没问题")
    return pointlist1,pointlist2,changepointlist#pl2为未知模型中的有问题的特征值位置下标

def save(pl1,pl2,path):
    x = np.random.rand(4096).reshape(64,64)
    for i in range(64):
        for j in range(64):
            x[i][j] = 0.15
    for i in pl1:
        x[int(i/64)][i%64] = 0.45
    for i in pl2:
        x[int(i/64)][i%64] = 1
        x[(int(i/64)+3)%64][(i+3)%64] = 1
        
# 创建颜色映射
    colors = ['green', 'yellow','red']
    cmap = ListedColormap(colors)

# 调整图片大小
    fig, ax = plt.subplots(figsize=(12, 12))

# 绘制热力图
    heatmap = sns.heatmap(x, annot=False, cmap=cmap, vmin=0, vmax=1, ax=ax, cbar=False, linewidths=0.5)
    plt.savefig(path)


if __name__ == '__main__':
    #调用说明:在已有normal的情况下分别对两个模型调用loadmodel
    #data需要一个dataloader的数据
    #pl1,pl2,changepointlist = get_similarty(model1,model2,data)
    #随后调用save(pl1,pl2,savepath)保存结果图
    
    debugmodeon()
    model1 = load_model(normal_path).eval()
    model2 = load_model(poision_path).eval()
    dataset = findmap(dataset_path)
    data = DataLoader(dataset, batch_size=1, shuffle=True)

    pl1,pl2,changepointlist = get_similarty(model1,model2,data)
    savepath = "../static/nodepic/node.png"
    save(pl1,pl2,savepath)