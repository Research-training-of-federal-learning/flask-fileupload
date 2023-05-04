import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from NeuralCleanse.data import get_data
from NeuralCleanse.mymodel.GTSRB import GTSRBNet
from torchvision import datasets, transforms
from NeuralCleanse.mymodel.vggface import VGGFace
from NeuralCleanse.mymodel.PUBFIG import PUBFIGNet

from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

import os
import shutil


def train(model, target_label, train_loader, param):
    print("Processing label: {}".format(target_label))

    width, height = param["image_size"]
    trigger = torch.rand((3, width, height), requires_grad=True)
    trigger = trigger.to(device).detach().requires_grad_(True)
    mask = torch.rand((width, height), requires_grad=True)
    mask = mask.to(device).detach().requires_grad_(True)

    Epochs = param["Epochs"]
    lamda = param["lamda"]

    min_norm = np.inf
    min_norm_count = 0

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam([{"params": trigger}, {"params": mask}], lr=0.005)
    model.to(device)
    model.eval()

    for epoch in range(Epochs):
        norm = 0.0
        # norm_classes = [0. for j in range(43)]
        for images, targets in tqdm.tqdm(train_loader, desc='Epoch %3d' % (epoch + 1)):
            optimizer.zero_grad()
            images = images.to(device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
            loss = criterion(y_pred, y_target) + lamda * torch.sum(torch.abs(mask))
            loss.backward()
            optimizer.step()

            # figure norm
            with torch.no_grad():
                # 防止trigger和norm越界
                torch.clip_(trigger, 0, 1)
                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))
                # norm_classes[] = torch.sum(torch.abs(mask))
        print("norm: {}".format(norm))

        # to early stop
        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1

        if min_norm_count > 30:
            break

    return trigger.cpu(), mask.cpu()


def reverse_engineer(param):
    # param = {
    #     "dataset": "PUBFIG",
    #     "Epochs": 1,
    #     "batch_size": 1,
    #     "lamda": 0.01,
    #     "num_classes": 83,
    #     "image_size": (224, 224)
    # }
    model = VGGFace(False).to(device)
    model.load_state_dict(torch.load('PUBFIG_model_last.pt.tar')['state_dict'])  # 导入数据
    # model = torch.load('model_cifar10.pkl').to(device)
    # _, _, x_test, y_test = get_data(param)
    # x_test, y_test = torch.from_numpy(x_test)/255., torch.from_numpy(y_test)
    # train_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=param["batch_size"], shuffle=False)
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.55206233, 0.44260582, 0.37644434), (0.2515312, 0.22786127, 0.22155665)),
    ])
    dataset = ImageFolder("../data/PUBFIG/pubfig83", transform=transform_test)
    train_dataset, test_dataset = random_split(dataset=dataset, lengths=[11070, 2733])
    train_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

    norm_list = []
    for label in range(param["num_classes"]):
        trigger, mask = train(model, label, train_loader, param)
        #trigger, mask = train(model, 6, train_loader, param)
        norm_list.append(mask.sum().item())

        trigger = trigger.cpu().detach().numpy()
        trigger = np.transpose(trigger, (1, 2, 0))
        plt.axis("off")
        plt.imshow(trigger)
        plt.savefig('NeuralCleanse/mask_PUBFIG/trigger_{}.png'.format(label), bbox_inches='tight', pad_inches=0.0)

        mask = mask.cpu().detach().numpy()
        plt.axis("off")
        plt.imshow(mask)
        plt.savefig('NeuralCleanse/mask_PUBFIG/mask_{}.png'.format(label), bbox_inches='tight', pad_inches=0.0)

    print(norm_list)
    min_label = np.argmin(norm_list)
    print(min_label)
    image_folder = "NeuralCleanse/mask_PUBFIG"
    # 遍历文件夹中的所有图片
    for filename in os.listdir(image_folder):
        # 如果图片文件名包含最小值对应的标签，则将该图片复制到名为 "min_label_images" 的文件夹中
        # print(filename)
        if "mask_" + str(min_label) in filename:
            # print("mask_"+str(min_label))
            shutil.copy(os.path.join(image_folder, filename), "NeuralCleanse/output_PUBFIG")
        if "trigger_" + str(min_label) in filename:
            # print("trigger_" + str(min_label))
            shutil.copy(os.path.join(image_folder, filename), "NeuralCleanse/output_PUBFIG")
    # 画图
    # 添加图形属性
    plt.xlabel('Label')
    plt.ylabel('L1-norm')
    plt.title('L1-norm range from Label')

    # norm_list=[92.30671691894531, 138.1082763671875, 77.82843017578125, 79.0025634765625,
    #            105.43791198730469, 79.33483123779297, 101.99564361572266, 98.91024017333984,
    #            12.002458572387695, 106.13054656982422]

    y = norm_list
    name_list = [str(x) for x in range(0,83) ]  # x轴标签
    # 计算前10%大的值
    sorted_y = sorted(y)
    threshold = sorted_y[int(param["num_classes"] * 0.1)]

    labels = [i for i, num in enumerate(norm_list) if num < threshold]
    sorted_labels = sorted(labels, key=lambda i: norm_list[i])
    # 将sorted_labels存储到txt文件中
    with open('NeuralCleanse/BackdoorLabel/PUBFIG_sorted_labels.txt', 'w') as f:
        for label in sorted_labels:
            f.write(str(label) + '\n')

    norms = [num for i, num in enumerate(norm_list) if num < threshold]
    sorted_norms = sorted(norms)
    with open('NeuralCleanse/BackdoorLabel/PUBFIG_sorted_norms.txt', 'w') as f:
        for norm in sorted_norms:
            f.write(str(norm) + '\n')

    for i in range(len(y)):
        if y[i] >= threshold:
            plt.bar(name_list[i], y[i], color='blue', width=0.35)
        else:
            plt.bar(name_list[i], y[i], color='red', width=0.35)
    # plt.bar(  # x=np.arange(10),  # 横坐标
    #     x=np.arange(param["num_classes"]),  # 横坐标
    #     height=y,  # 柱状高度
    #     width=0.35,  # 柱状宽度
    #     # label='小明',  # 标签
    #     edgecolor='k',  # 边框颜色
    #     color='r',  # 柱状图颜色
    #     tick_label=name_list,  # 每个柱状图的坐标标签
    #     linewidth=3)  # 柱状图边框宽度
    # plt.legend()  # 显示标签
    # plt.show()
    # 图片的显示及存储
    # plt.show()   #这个是图片显示
    plt.savefig('NeuralCleanse/output_PUBFIG/normDistribution.png')  # 图片的存储
    plt.close()  # 关闭matplotlib


if __name__ == "__main__":
    print("gpu is available:")
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reverse_engineer()
