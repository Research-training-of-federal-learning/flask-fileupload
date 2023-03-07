# -*- coding:utf-8 -*-
import os
import sys
import torch
import shutil
from nturl2path import pathname2url


#使用方法:代码与文件同一路径， python poly.py n file_1 file_2 …… file_n

# def controller(n,mypy,myname,myparams,mycommit):
#     shutil.rmtree("saved_models")
#     os.mkdir("saved_models")


#     for i in range(0,n):
#         os.system(mypy + " training.py --name " + myname[i] + " --params " + myparams[i] + " --commit " + mycommit[i])
    


#     #扫描已学习文件
#     path = []
#     for root, dirs, files in os.walk("saved_models"):
#         for mydir in dirs:
#             path.append("saved_models/"+mydir+"/model_last.pt.tar.best")
#             #print(mydir+"/model_last.pt.tar.best")
#     #print(path)
#     model_add(n,path,'cpu')
#     print("模型已平均聚合")


def model_add(n,file_name,device='cpu'):
    for i in range(0,n):
        if(not os.path.exists(file_name[i])):
            print(f"{file_name[i]}找不到")
            return
    if(n == 1):
        print("无需聚合")
        return
    else:
        model = torch.load(file_name[0], map_location=torch.device(device))
    for i in range(1,n):
        model2 = torch.load(file_name[i],map_location=torch.device(device))
        model['state_dict']['conv1.weight']+=model2['state_dict']['conv1.weight']
        model['state_dict']['conv1.bias']+=model2['state_dict']['conv1.bias']
        model['state_dict']['conv2.weight']+=model2['state_dict']['conv2.weight']
        model['state_dict']['conv2.bias']+=model2['state_dict']['conv2.bias']
        model['state_dict']['fc1.weight']+=model2['state_dict']['fc1.weight']
        model['state_dict']['fc1.bias']+=model2['state_dict']['fc1.bias']
        model['state_dict']['fc2.weight']+=model2['state_dict']['fc2.weight']
        model['state_dict']['fc2.bias']+=model2['state_dict']['fc2.bias']
    model['state_dict']['conv1.weight']/=n
    model['state_dict']['conv1.bias']/=n
    model['state_dict']['conv2.weight']/=n
    model['state_dict']['conv2.bias']/=n
    model['state_dict']['fc1.weight']/=n
    model['state_dict']['fc1.bias']/=n
    model['state_dict']['fc2.weight']/=n
    model['state_dict']['fc2.bias']/=n
    torch.save(model, "./saved_models/model/lastpoly.pt.tar")


if __name__ == "__main__":
    file_name=["saved_models/model_MNIST_Oct.21_14.55.26_mnist/model_last.pt.tar","saved_models/input/model_last.pt.tar"]
    model_add(2,file_name,"cpu")
    # myname=['mnist','mnist']
    # myparams=['configs/mnist_params.yaml','configs/mnist_params.yaml']
    # mycommit=['none','none']
    # contaoller(2,"python37",myname,myparams,mycommit)


