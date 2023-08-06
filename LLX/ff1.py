import numpy as np
import sys
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,DataLoader,Subset
import torch.nn.functional as F
import torch
import torch.nn as nn
from LLX.vggface import VGGFace
from sklearn.neighbors import KNeighborsClassifier
from math import log



#poison会小一些
# model_path = sys.argv[1]



#PCA
def PCA_svd(X, k=100, center=False):
    n = X.size()[0]
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    H = H.cuda()
    X_center =  torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components  = v[:k].t()
    #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return components





#读图片
def findmap():
    input_data_path = "../PUBFIG/pubfig83"
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

def main():
    dataset = findmap()



    #预测

    # sub_dataset = Subset(dataset,range(85))
    model_path = "../pre_models/GTSRB/test.pth"
    #读模型
    use_cuda = True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    # device = "cpu"
    model = VGGFace(False).to(device)

    #为网络导入参数
    model.load_state_dict(torch.load(model_path)['state_dict'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()
    reallist1 = []
    reallist2 = []
    reallist3 = []
    tarlist = []
    num = 0
    for batch in dataloader:
        if num >= 50:
            break
        else:
            input_data, target = batch
            tarlist.append(target.item())
            input_data = input_data.to(device)
            output,list = model(input_data)
            del list[0]
            del list[0]
            del list[1]
            del list[1]
            #list是最后三层的激活值
            k = 0
            for l in list:
                if k%3 == 0:
                    reallist1.append(PCA_svd(l))
                    k+=1
                elif k%3 == 1:
                    reallist2.append(PCA_svd(l))
                    k+=1
                else:
                    reallist3.append(PCA_svd(l))
                    k=0
            num+=1

    #KNN
    reallist11 = []
    for i in reallist1:
        reallist11.append(i.cpu().detach().numpy()[0].tolist())
    reallist22 = []
    for i in reallist2:
        reallist22.append(i.cpu().detach().numpy()[0].tolist())
    reallist33=[]
    for i in reallist3:
        reallist33.append(i.cpu().detach().numpy()[0].tolist())
        

    #KNN
    knn1 = KNeighborsClassifier(n_neighbors=5)
    knn1.fit(reallist11,tarlist)
    knn2 = KNeighborsClassifier(n_neighbors=5)
    knn2.fit(reallist22,tarlist)
    knn3 = KNeighborsClassifier(n_neighbors=5)
    knn3.fit(reallist33,tarlist)

    #predict
    realtar1 = []
    realtar2 = []
    realtar3 = []

    realtar1.append(knn1.predict(reallist11).tolist())
    realtar2.append(knn2.predict(reallist22).tolist())
    realtar3.append(knn3.predict(reallist33).tolist())


    #求p
    cnt_change1 = 0
    cnt_change2 = 0
    for i in range(len(realtar1[0])):
        if realtar1[0][i] != realtar2[0][i]:
            cnt_change1+=1
        if realtar2[0][i] != realtar3[0][i]:
            cnt_change2+=1

    p1_c = log(cnt_change1/len(realtar1[0]))
    p2_c = log(cnt_change2/len(realtar1[0]))
    p1_uc = 1-p1_c
    p2_uc = 1-p2_c
    #求τ
    LLX = 0
    for i in range(len(realtar1[0])):
        if realtar1[0][i] != realtar2[0][i]:
            LLX+=p1_c
        else:
            LLX+=p2_uc
        if realtar2[0][i] != realtar3[0][i]:
            LLX+=p1_c
        else:
            LLX+=p2_uc
    return LLX