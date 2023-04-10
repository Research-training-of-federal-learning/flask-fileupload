from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from reverse.MNIST.model import Model
import sys



weight = [[0. for j in range(28)] for i in range(28)]



weight=torch.Tensor(weight)
device = torch.device("cuda" if (True and torch.cuda.is_available()) else "cpu")
device = "cpu"
weight=weight.to(device)



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
	print("")
	print("")
	for i in range(len(lead)):
		for j in range(len(lead)):
			if(i!=j):
				print(str(lead[i][j]),end=" ")
			else:
				print(str(0),end=" ")
		print("")


def test(model,device,test_loader,epsilon,ep):#测试函数
	lead =[[[0 for j in range(10)] for i in range(10)] for k in range(10)]
	correct=0#存放正确的个数
	adv_examples=[]#存放正确的例子
	adv_examples=[]#存放正确的例子
	test_loader_lenth = len(test_loader)
	no=0
	for data,target in test_loader:
		no+=1
		if(no==100):
			print(100)
			#break
		print("\r", end="")
		print("进度: {}%: ".format(100*no//test_loader_lenth), end="")
		sys.stdout.flush()
		data,target=data.to(device),target.to(device)
		data.requires_grad=True


		output=model(data)

		init_pred=output.max(1,keepdim=True)[1]#选取最大的类别概率
		if init_pred.item()!=target.item():#判断类别是否相等
			continue
		else:
			init_pred[0] = second(output)
		##
		#perturbed_data=data
		##
		for j in range(10):
			if(j==target.item()):
				continue
			##
			perturbed_data=data.clone().detach()
			perturbed_data=perturbed_data.to(device)
			perturbed_data.requires_grad=True
			output=model(perturbed_data)
			if(j==0):
				#print(output)
				init1=output.clone()
			##
			init_pred[0]=j

			for i in range(ep):
				loss=F.nll_loss(output,init_pred[0])
				model.zero_grad()
				loss.backward()
				data_grad=perturbed_data.grad.data
				perturbed_data=fgsm_attack(perturbed_data,epsilon,data_grad)
				perturbed_data=perturbed_data.detach()
				perturbed_data.requires_grad=True
				output=model(perturbed_data)
				final_pred=output.max(1,keepdim=True)[1]
				lead[i][target.item()][final_pred.item()]+=1


	final_acc = correct / float(len(test_loader))#算正确率


	# Return the accuracy and an adversarial example
	return final_acc, adv_examples, lead

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
		for j in range(len(statistics_sum)):
			if(statistics_sum[j]==0):
				statistics_result_d[j]=-1
				statistics_sum[j]+=1

		statistics_check[i]=float(statistics_result[i])/float(statistics_sum[i])

		for j in range(len(totle_leads[i])):
			if((statistics_result[i]+totle_leads[i][j])!=0):
				t = statistics_result[i]/(statistics_result[i]+totle_leads[i][j])
			if(t<statistics_result_t[i] and i!=j and j!=statistics_result_d[i]):
				statistics_result_t[i]=t
			#print(j,statistics_result_d[i],totle_leads[i][j],statistics_result[i],statistics_result_t[i],t)
		if(statistics_check[i]<(2.0/len(totle_leads)) or statistics_result_t[i]<=0.5):
			statistics_result_d[i]=-1

		#print(statistics_result[i],statistics_sum[i],statistics_check[i],statistics_result_t[i],statistics_result_d[i])




def find(ep,pretrained_model,use_cuda,epsilons):


	totle_leads=[[0 for j in range(10)] for i in range(10)]
	totle_leads=np.array(totle_leads)
	test_loader = torch.utils.data.DataLoader(#导入数据
	datasets.MNIST('.data', train=False, download=True, transform=transforms.Compose([
			transforms.ToTensor(),
			])),
		batch_size=1, shuffle=True)
	# D选择使用cpu或者是gpu
	device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
	device = "cpu"


	# 初始化网络
	model = Net(10).to(device)


	model.load_state_dict(torch.load(pretrained_model, map_location='cpu')['state_dict'])#导入数据

	# 进入测试模式
	model.eval()

	accuracies = []
	examples = []

	# Run test for each epsilon
	for i in range(1):
		for eps in epsilons:
			acc, ex, lead = test(model, device, test_loader, eps,ep)
			totle_leads = totle_leads + np.array(lead)
			if(i!=0):
				continue
			accuracies.append(acc)
			examples.append(ex)

	for i in range(len(totle_leads)):
		for j in range(len(totle_leads[i])):
			for k in range(len(totle_leads[i][j])):
				if(j==k):
					totle_leads[i][j][k]=0

	return totle_leads.tolist()


if __name__ == '__main__':
	pretrained_model = "lenet_mnist_model.pth"
	use_cuda=True
	epsilons = [0, .05, .1, .15, .2, .25, .3]
	epsilons = [0.05]
	find(10,pretrained_model,use_cuda,epsilons)