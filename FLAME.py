import numpy as np
import random
import torch


# 转换为np数组
def torchfile2np(to):
    n = torch.load(to, map_location=torch.device("cpu"))['state_dict']
    key_list = list(n.keys())
    mylist = np.arange(0)
    for i in range(len(key_list)):
        mylist = np.append(mylist, n[key_list[i]].view(-1).numpy())
    return mylist

def data_needed(filePath):
    import os  # 引入os
    file_name = list()  # 新建列表
    for i in os.listdir(filePath):  # 获取filePath路径下所有文件名
        data_collect = ''.join(i)  # 文件名字符串格式
        file_name.append(data_collect)  # 将文件名作为列表元素填入
    return (file_name)  # 返回列表


def np2model(mylist, model):
    mymodel = mylist[0:1728]
    model['state_dict']['features.conv_1_1.weight'] = torch.tensor(mymodel).view([64, 3, 3, 3])
    mymodel = mylist[1728:1792]
    model['state_dict']['features.conv_1_1.bias'] = torch.tensor(mymodel).view([64])
    mymodel = mylist[1792:38656]
    model['state_dict']['features.conv_1_2.weight'] = torch.tensor(mymodel).view([64, 64, 3, 3])
    mymodel = mylist[38656:38720]
    model['state_dict']['features.conv_1_2.bias'] = torch.tensor(mymodel).view([64])
    mymodel = mylist[38720:112448]
    model['state_dict']['features.conv_2_1.weight'] = torch.tensor(mymodel).view([128, 64, 3, 3])
    mymodel = mylist[112448:112576]
    model['state_dict']['features.conv_2_1.bias'] = torch.tensor(mymodel).view([128])
    mymodel = mylist[112576:260032]
    model['state_dict']['features.conv_2_2.weight'] = torch.tensor(mymodel).view([128, 128, 3, 3])
    mymodel = mylist[260032:260160]
    model['state_dict']['features.conv_2_2.bias'] = torch.tensor(mymodel).view([128])
    mymodel = mylist[260160:555072]
    model['state_dict']['features.conv_3_1.weight'] = torch.tensor(mymodel).view([256, 128, 3, 3])
    mymodel = mylist[555072:555328]
    model['state_dict']['features.conv_3_1.bias'] = torch.tensor(mymodel).view([256])
    mymodel = mylist[555328:1145152]
    model['state_dict']['features.conv_3_2.weight'] = torch.tensor(mymodel).view([256, 256, 3, 3])
    mymodel = mylist[1145152:1145408]
    model['state_dict']['features.conv_3_2.bias'] = torch.tensor(mymodel).view([256])
    mymodel = mylist[1145408:1735232]
    model['state_dict']['features.conv_3_3.weight'] = torch.tensor(mymodel).view([256, 256, 3, 3])
    mymodel = mylist[1735232:1735488]
    model['state_dict']['features.conv_3_3.bias'] = torch.tensor(mymodel).view([256])
    mymodel = mylist[1735488:2915136]
    model['state_dict']['features.conv_4_1.weight'] = torch.tensor(mymodel).view([512, 256, 3, 3])
    mymodel = mylist[2915136:2915648]
    model['state_dict']['features.conv_4_1.bias'] = torch.tensor(mymodel).view([512])
    mymodel = mylist[2915648:5274944]
    model['state_dict']['features.conv_4_2.weight'] = torch.tensor(mymodel).view([512, 512, 3, 3])
    mymodel = mylist[5274944:5275456]
    model['state_dict']['features.conv_4_2.bias'] = torch.tensor(mymodel).view([512])
    mymodel = mylist[5275456:7634752]
    model['state_dict']['features.conv_4_3.weight'] = torch.tensor(mymodel).view([512, 512, 3, 3])
    mymodel = mylist[7634752:7635264]
    model['state_dict']['features.conv_4_3.bias'] = torch.tensor(mymodel).view([512])
    mymodel = mylist[7635264:9994560]
    model['state_dict']['features.conv_5_1.weight'] = torch.tensor(mymodel).view([512, 512, 3, 3])
    mymodel = mylist[9994560:9995072]
    model['state_dict']['features.conv_5_1.bias'] = torch.tensor(mymodel).view([512])
    mymodel = mylist[9995072:12354368]
    model['state_dict']['features.conv_5_2.weight'] = torch.tensor(mymodel).view([512, 512, 3, 3])
    mymodel = mylist[12354368:12354880]
    model['state_dict']['features.conv_5_2.bias'] = torch.tensor(mymodel).view([512])
    mymodel = mylist[12354880:14714176]
    model['state_dict']['features.conv_5_3.weight'] = torch.tensor(mymodel).view([512, 512, 3, 3])
    mymodel = mylist[14714176:14714688]
    model['state_dict']['features.conv_5_3.bias'] = torch.tensor(mymodel).view([512])
    mymodel = mylist[14714688:117475136]
    model['state_dict']['fc.fc6.weight'] = torch.tensor(mymodel).view([4096, 25088])
    mymodel = mylist[117475136:117479232]
    model['state_dict']['fc.fc6.bias'] = torch.tensor(mymodel).view([4096])
    mymodel = mylist[117479232:134256448]
    model['state_dict']['fc.fc7.weight'] = torch.tensor(mymodel).view([4096, 4096])
    mymodel = mylist[134256448:134260544]
    model['state_dict']['fc.fc7.bias'] = torch.tensor(mymodel).view([4096])
    mymodel = mylist[134260544:145000256]
    model['state_dict']['fc.fc8.weight'] = torch.tensor(mymodel).view([2622, 4096])
    mymodel = mylist[145000256:145002878]
    model['state_dict']['fc.fc8.bias'] = torch.tensor(mymodel).view([2622])
    return model


# 取最小值
def Min(a, b):
    if a > b:
        return a
    else:
        return b


# 余弦距离
def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similarity = np.dot(a, b.T) / (a_norm * b_norm.T)
    dist = 1. - similarity
    return dist


# 欧几里得距离
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def find_eps(distance_D, eps):
    '''找到距离≤eps的样本的索引
    :param distance_D(mat):样本i与其他样本之间的距离
    :param eps(float):半径的大小
    :return: ind(list):与样本i之间的距离≤eps的样本的索引
    '''
    ind = []
    n = np.shape(distance_D)[1]
    for j in range(n):
        if distance_D[0, j] <= eps:
            ind.append(j)
    return ind


class FLAME:
    def __init__(self, n, size, database, G0):
        self.model = torch.load(database[0], map_location=torch.device("cpu"))
        self.S = None  # 存放欧几里得中值
        self.c = None  # 存放余弦距离
        self.b = None  # 存放聚类后结果
        self.e = None  # 存放欧几里得距离
        self.n = n  # n是客户端数量
        self.size = size  # size是模型的参数个数
        self.L = n  # L是聚类后允许的参数
        self.W = []  # W列表存储客户端更新后的参数
        self.newW = None  # 裁剪后的W
        self.G0 = G0  # self.G0 = np.zeros([self.size], dtype=int)  # 上一轮参数
        self.G = np.zeros([self.size], dtype=float)  # 聚合后新的参数
        self.Lambda = 0.001  # 噪声参数，0.001for IC NLP,0.01for NIDS
        # for i in range(self.size):
        #     self.G0[i] = random.randint(0, 100)
        for i in range(n):
            self.W.append(torchfile2np(database[i]))

    # 计算W之间的余弦距离
    def get_cij(self):
        self.c = np.mat(np.zeros((self.n, self.n)))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.c[i, j] = cosine_distance(self.W[i], self.W[j])
                self.c[j, i] = self.c[i, j]

    # 聚类
    def dbscan(self, eps, MinPts):
        '''DBSCAN算法
        :param data(mat):需要聚类的数据集
        :param eps(float):半径
        :param MinPts(int):半径内最少的数据点数
        :return:
            types(mat):每个样本的类型：核心点、边界点、噪音点
            sub_class(mat):每个样本所属的类别
        '''
        data = self.W
        m = np.shape(data)[0]
        # 在types中，1为核心点，0为边界点，-1为噪音点
        types = np.mat(np.zeros((1, m)))
        sub_class = np.mat(np.zeros((1, m)))
        # 用于判断该点是否处理过，0表示未处理过
        dealt = np.mat(np.zeros((m, 1)))
        # 计算每个数据点之间的距离
        dis = self.c
        # 用于标记类别
        number = 1

        # 对每一个点进行处理
        for i in range(m):
            # 找到未处理的点
            if dealt[i, 0] == 0:
                # 找到第i个点到其他所有点的距离
                D = dis[i,]
                # 找到半径eps内的所有点
                ind = find_eps(D, eps)
                # 区分点的类型
                # 边界点
                if len(ind) > 1 and len(ind) < MinPts + 1:
                    types[0, i] = 0
                    sub_class[0, i] = 0
                # 噪音点
                if len(ind) == 1:
                    types[0, i] = -1
                    sub_class[0, i] = -1
                    dealt[i, 0] = 1
                # 核心点
                if len(ind) >= MinPts + 1:
                    types[0, i] = 1
                    for x in ind:
                        sub_class[0, x] = number
                    # 判断核心点是否密度可达
                    while len(ind) > 0:
                        dealt[ind[0], 0] = 1
                        D = dis[ind[0],]
                        tmp = ind[0]
                        del ind[0]
                        ind_1 = find_eps(D, eps)

                        if len(ind_1) > 1:  # 处理非噪音点
                            for x1 in ind_1:
                                sub_class[0, x1] = number
                            if len(ind_1) >= MinPts + 1:
                                types[0, tmp] = 1
                            else:
                                types[0, tmp] = 0

                            for j in range(len(ind_1)):
                                if dealt[ind_1[j], 0] == 0:
                                    dealt[ind_1[j], 0] = 1
                                    ind.append(ind_1[j])
                                    sub_class[0, ind_1[j]] = number
                    number += 1

        # 最后处理所有未分类的点为噪音点
        ind_2 = ((sub_class == 0).nonzero())[1]
        for x in ind_2:
            sub_class[0, x] = -1
            types[0, x] = -1

        return types, sub_class

    def clustering(self):
        types, sub_class = self.dbscan(3, 1)
        self.b = []
        cnt = 0
        types = np.asarray(types)
        types = types[0]
        for i in types:
            if i == 1:
                self.b.append(cnt)
            cnt += 1
        self.L = cnt

    # 获得与上个模型的欧几里得距离
    def get_ei(self):
        self.e = []
        for i in range(self.n):
            self.e.append(euclidean_distance(self.G0, self.W[i]))

    # 获得欧几里得中值
    def median(self):
        self.S = 0
        for i in range(self.n):
            self.S += self.e[i]
        self.S = self.S / self.n

    # 裁剪
    def clipping(self):
        self.newW = []
        for i in range(self.L):
            self.newW.append(self.G0 + (self.W[self.b[i]] - self.G0) * Min(1, self.S / self.e[self.b[i]]))

    # 更新出新的W
    def update(self):
        self.get_cij()
        self.clustering()
        self.get_ei()
        self.median()
        self.clipping()
        for i in range(self.L):
            self.G += self.newW[i]
        self.G = self.G / self.L
        sigma = self.Lambda * self.S
        self.G = self.G
        for i in range(self.size):
            self.G[i] += random.gauss(0, sigma)

    # 返回新的G
    def get_G(self):
        return np2model(self.G, self.model)
