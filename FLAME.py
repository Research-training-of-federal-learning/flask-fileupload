import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib;

matplotlib.use('tkagg')


# 转换为np数组
def torchfile2np(to):
    n = torch.load(to, map_location=torch.device("cpu"))['state_dict']
    key_list = list(n.keys())
    mylist = np.arange(0)
    for i in range(len(key_list) - 2, len(key_list)):
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
    mymodel = mylist[0:10739712]
    model['state_dict']['fc.fc8.weight'] = torch.tensor(mymodel).view([2622, 4096])
    mymodel = mylist[10739712:10742334]
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
    def __init__(self, n, size, database, G0_file_path):
        self.model = torch.load(G0_file_path, map_location=torch.device("cpu"))
        self.S = None  # 存放欧几里得中值
        self.c = None  # 存放余弦距离
        self.b = None  # 存放聚类后结果
        self.e = None  # 存放欧几里得距离
        self.database = database
        self.sinlevel = [0] * n
        self.n = n  # n是客户端数量
        self.size = size  # size是模型的参数个数
        self.L = n  # L是聚类后允许的参数
        self.W = []  # W列表存储客户端更新后的参数
        self.newW = None  # 裁剪后的W
        self.G0 = torchfile2np(G0_file_path)  # self.G0 = np.zeros([self.size], dtype=int)  # 上一轮参数
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
        types, sub_class = self.dbscan(0.0005, 5)
        self.b = []
        cnt = 0
        types = np.asarray(types)
        types = types[0]
        for i in types:
            if i == 1:
                self.b.append(cnt)
                self.sinlevel[cnt] = 1
            cnt += 1
        self.L = len(self.b)

    # 获得与上个模型的欧几里得距离
    def get_ei(self):
        self.e = []
        for i in range(self.L):
            self.e.append(euclidean_distance(self.G0, self.W[self.b[i]]))

    # 获得欧几里得中值
    def median(self):
        self.S = 0
        for i in range(self.L):
            self.S += self.e[self.b[i]]
        self.S = self.S / self.L

    # 裁剪
    def clipping(self):
        self.newW = []
        for i in range(self.L):
            k = min(1, self.S / self.e[self.b[i]])
            self.newW.append(self.G0 + (self.W[self.b[i]] - self.G0) * k)
            self.sinlevel[self.b[i]] = k

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
        for i in range(self.size):
            self.G[i] += random.gauss(0, sigma)

    # 返回新的G
    def get_G(self):
        return np2model(self.G, self.model)

    # 绘制不信任度图
    def draw_sinlevel(self):
        drawfile = """<!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>The degree of trust of the model in the aggregation</title>
            <script src="https://cdn.staticfile.org/echarts/4.3.0/echarts.min.js"></script>
        </head>
        <body>
            <div id="main" style="width: 800px;height:600px;"></div>
            <script type="text/javascript">
                var chartDom = document.getElementById('main');
        var myChart = echarts.init(chartDom);
        var option;

        option = {
          title: {
            text: 'Model trust ratio',
            subtext: 'present',
            left: 'center'
          },
          tooltip: {
            trigger: 'item'
          },
          legend: {
            orient: 'vertical',
            left: 'left'
          },
          series: [
            {
              name: 'Access From',
              type: 'pie',
              radius: '50%',
              data: [ 
        """
        drawfile2 = """
              ],
              emphasis: {
                itemStyle: {
                  shadowBlur: 10,
                  shadowOffsetX: 0,
                  shadowColor: 'rgba(0, 0, 0, 0.5)'
                }
              }
            }
          ]
        };
        option && myChart.setOption(option);
            </script>
        </body>
        </html>"""
        drawdata = "\t{ value: %f, name: '%s' },\n"
        filedata = drawfile
        for i in range(len(self.sinlevel)):
            k = self.database[i].find("\\")
            name = self.database[i][k + 1:]
            filedata += drawdata % (self.sinlevel[i], name)
        filedata += drawfile2
        print(1111)
        f = open("templates\\Trust_degree.html", "w")
        f.write(filedata)
        f.close()

    # 绘制不信任图二：
    def draw_sinlevel2(self):
        plt.rcParams["font.sans-serif"] = ['SimHei']
        plt.rcParams["axes.unicode_minus"] = False

        for i in range(len(self.sinlevel)):
            k = self.database[i].find("\\")
            name = self.database[i][k + 1:]
            m = 1 - self.sinlevel[i]
            if m == 0:
                continue
            plt.bar(name, m)

        plt.title("The degree of trust of the model in the aggregation")
        plt.xlabel("models")
        plt.ylabel("Model trust ratio")
        plt.savefig('templates/Trust_degree.png')
        plt.close()

    def draw_level3(self):
        y_data = []
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        for i in range(self.n):
            m = np.absolute(self.W[i] - self.G0)
            y = np.hsplit(m, 4097)
            ydatas = np.mean(y, axis=1)
            y_data.append(ydatas)
        '''plot starting ... '''
        number_of_point = 241
        x_data = range(0, number_of_point)
        fig = plt.figure()
        plt.rcParams['savefig.dpi'] = 4096  # 图片像素
        plt.rcParams['figure.dpi'] = 4096  # 分辨率
        ax = fig.add_subplot(111, projection='3d')
        labels = []
        picle = 1  # 参照组
        group_num = 1
        for i in range(len(y_data)):  # k<0.9显示
            if picle == 1 and self.sinlevel[i]:
                label = "model_reference"
                labels.append(label)
                c = "#" + str(hex(random.randint(0x111111, 0xffffff)))[2:]
                for j in range(17):
                    k = float(j) / 17
                    t = y_data[i][j * number_of_point:(j + 1) * number_of_point]
                    ax.scatter(xs=x_data, ys=k, zs=t, c=c, s=1, alpha=1, label=label, marker='o')
                picle = 0
            if self.sinlevel[i] < 0.9:
                label = "model" + str(i)
                labels.append(label)
                c = "#" + str(hex(random.randint(0x111111, 0xffffff)))[2:]
                for j in range(17):
                    k = float(j) / 17
                    t = y_data[i][j * number_of_point:(j + 1) * number_of_point]
                    ax.scatter(xs=x_data, ys=group_num + k, zs=t, c=c, s=1, alpha=1, label=label, marker='o')
                group_num += 1
        yticks = []
        for i in range(len(labels)):
            yticks.append(i)
        ax.set_xticklabels([" ", " ", "layer", " ", " "], fontsize=20)
        ax.set_yticklabels(labels, fontsize=20)
        ax.set_zlabel('distance', fontsize=16)
        ax.set_xticks([0, 50, 100, 1500, 200, 250])  # x 轴刻度密度
        ax.set_yticks(yticks)  # y 轴刻度密度
        ax.set_xlim(left=0, right=number_of_point)  # x 轴显示范围
        ax.set_ylim(bottom=0, top=len(labels))  # y 轴显示范围
        plt.tick_params(labelsize=13)  # 刻度字体大小
        # plt.savefig('student_score.pdf')
        plt.show()

    def draw_level4(self):
        y_data = []
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        for i in range(self.n):
            m = np.absolute(self.W[i] - self.G0)
            y = np.hsplit(m, 4097)
            ydatas = np.max(y, axis=1)
            y_data.append(ydatas)
        '''plot starting ... '''
        number_of_point = 241
        x_data = range(0, number_of_point)
        fig = plt.figure()
        # plt.rcParams['savefig.dpi'] = 4096  # 图片像素
        # plt.rcParams['figure.dpi'] = 4096  # 分辨率
        ax = fig.add_subplot(111, projection='3d')
        labels = []
        picle = 1  # 参照组
        group_num = 1
        for i in range(len(y_data)):  # k<0.9显示
            if picle == 1 and self.sinlevel[i]:
                label = "model_reference"
                labels.append(label)
                c = "#" + str(hex(random.randint(0x111111, 0xffffff)))[2:]
                for j in range(17):
                    k = float(j) / 17
                    t = y_data[i][j * number_of_point:(j + 1) * number_of_point]
                    ax.scatter(xs=x_data, ys=k, zs=t, c=c, s=1, alpha=1, label=label, marker='o')
                picle = 0
            if self.sinlevel[i] < 0.9:
                label = "model" + str(i)
                labels.append(label)
                c = "#" + str(hex(random.randint(0x111111, 0xffffff)))[2:]
                for j in range(17):
                    k = float(j) / 17
                    t = y_data[i][j * number_of_point:(j + 1) * number_of_point]
                    ax.scatter(xs=x_data, ys=group_num + k, zs=t, c=c, s=1, alpha=1, label=label, marker='o')
                group_num += 1
        yticks = []
        for i in range(len(labels)):
            yticks.append(i)
        ax.set_xticklabels([" ", " ", "layer", " ", " "], fontsize=20)
        ax.set_yticklabels(labels, fontsize=20)
        ax.set_zlabel('distance', fontsize=16)
        ax.set_xticks([0, 50, 100, 1500, 200, 250])  # x 轴刻度密度
        ax.set_yticks(yticks)  # y 轴刻度密度
        ax.set_xlim(left=0, right=number_of_point)  # x 轴显示范围
        ax.set_ylim(bottom=0, top=len(labels))  # y 轴显示范围
        plt.tick_params(labelsize=13)  # 刻度字体大小
        # plt.savefig('student_score.pdf')
        plt.show()

    # 返回相关值
    def get_sinlevel(self):
        return self.sinlevel


# 节点修复功能
def fix_model(model_path, model_G0_path, matrix, way):
    base_model = torch.load(model_path, map_location=torch.device("cpu"))
    model_matrix = torchfile2np(model_path)[0:10739712]

    if way == 1:
        matrix = matrix * (-1) + 1
        model_matrix = model_matrix * matrix

    elif way == 2:
        matrix1 = matrix * (-1) + 1
        model_matrix1 = model_matrix * matrix1
        matrix2 = matrix * (-1)
        model_matrix2 = model_matrix * matrix2
        model_matrix = model_matrix1 + model_matrix2

    elif way == 3:
        model_G0 = torchfile2np(model_G0_path)[0:10739712]
        model_G0 = model_G0 * matrix
        matrix = matrix * (-1) + 1
        model_matrix = model_matrix * matrix
        model_matrix = model_matrix + model_G0

    base_model['state_dict']['fc.fc8.weight'] = torch.tensor(model_matrix).view([2622, 4096])
    return base_model

