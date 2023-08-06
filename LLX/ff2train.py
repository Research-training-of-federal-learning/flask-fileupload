
#正常的符合度会大一点


import numpy as np
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,DataLoader,Subset
import torch
from LLX.vggface import VGGFace


device = "cuda"
input_data_path = "../PUBFIG/pubfig83"
model_path2 = "../pre_models/GTSRB/test.pth"
model_path1 = "../pre_models/GTSRB/pre_train.pth"


RES_LEN = 500

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

def train(input_data_path,model_path1,model_path2):
    datasets = findmap(input_data_path)
    model1 = load_model(model_path1)
    model1.eval()
    model2 = load_model(model_path2)
    model2.eval()
    
    dataloader = DataLoader(datasets, batch_size=1, shuffle=True)
    model1_X = []
    model2_X = []
    cnt = 0
    for i in dataloader:
        if cnt > 5:
            break
        else:
            input_data,target = i
            input_data = input_data.to(device)
            output1,list1 = model1(input_data)
            model1_X.append(list1[0][-1].tolist())
            output2,list2 = model2(input_data)
            model2_X.append(list2[0][-1].tolist())
            cnt += 1
    #X下两个模型的最后一层激活值拿到了
    return model1_X,model2_X

def divide(x,y):
    for i in range(0,len(x)):
        x[i]-=y[i]
        return x

def kernel(x,y,theta):
    #theta为超参数
    # print(len(x),len(y))
    return np.exp(-np.linalg.norm(divide(x,y))**2/(theta**2))


def test(model1_X,model2_X,model_path1,model_path2):
    model1 = load_model(model_path1)
    model2 = load_model(model_path2)
    model1.eval()
    model2.eval()
    datasets = findmap(input_data_path)
    dataloader = DataLoader(datasets, batch_size=1, shuffle=True)
    input,target = next(iter(dataloader))
    input = input.to(device)
    output1,x1 = model1(input)
    output2,x2 = model2(input)
    ans1 = 0
    ans2 = 0
    x11 = x1[0][-1].tolist()
    x22 = x2[0][-1].tolist()
    x11_ = []
    x22_ = []
    for i in x11:
        if i>0:
            x11_.append(i)
    for i in x22:
        if i > 0:
            x22_.append(i)
    x11_ = x11_[0:RES_LEN]
    x22_ = x22_[0:RES_LEN]
    for i in model1_X:
        k = []
        for j in i:
            if j >0:
                k.append(j)
        k = k[0:RES_LEN]
        ans1 += kernel(k,x11_,50)

    for i in model2_X:
        k = []
        for j in i:
            if j >0:
                k.append(j)
        k = k[0:RES_LEN]
        ans2 += kernel(k,x22_,50)
    return (ans1+ans2)*1e3

def main():
    modelX1,modelX2 = train(input_data_path,model_path1,model_path2)
    return test(modelX1,modelX2,model_path1,model_path1)