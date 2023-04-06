import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from NeuralCleanse.data import get_data
from NeuralCleanse.mymodel.GTSRB import GTSRBNet
from torchvision import datasets, transforms


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


def reverse_engineer():
    param = {
        "dataset": "GTSRB",
        "Epochs": 10,
        "batch_size": 64,
        "lamda": 0.01,
        "num_classes": 43,
        "image_size": (32, 32)
    }
    classes = (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42)
    model = GTSRBNet(num_classes=len(classes)).to(device)#将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行
    model.load_state_dict(torch.load('model_last.pt.tar')['state_dict'])  # 导入数据
    # model = torch.load('model_cifar10.pkl').to(device)
    # _, _, x_test, y_test = get_data(param)
    # x_test, y_test = torch.from_numpy(x_test)/255., torch.from_numpy(y_test)
    # train_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=param["batch_size"], shuffle=False)
    train_loader = torch.utils.data.DataLoader(  # 导入数据
        datasets.GTSRB(root='../data', split="test", download=True,#向上两级文件夹
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)), transforms.ToTensor(),
                           transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),#数据集的平均数和方差
                       ])),
        batch_size=param["batch_size"],
        shuffle=False,  # 在每次迭代训练时否将数据洗牌
        num_workers=0)

    norm_list = []
    for label in range(param["num_classes"]):
        trigger, mask = train(model, label, train_loader, param)
        # trigger, mask = train(model, 8, train_loader, param)
        norm_list.append(mask.sum().item())

        trigger = trigger.cpu().detach().numpy()
        trigger = np.transpose(trigger, (1, 2, 0))
        plt.axis("off")
        plt.imshow(trigger)
        plt.savefig('NeuralCleanse/mask_GTSRB/trigger_{}.png'.format(label), bbox_inches='tight', pad_inches=0.0)

        mask = mask.cpu().detach().numpy()
        plt.axis("off")
        plt.imshow(mask)
        plt.savefig('NeuralCleanse/mask_GTSRB/mask_{}.png'.format(label), bbox_inches='tight', pad_inches=0.0)

    print(norm_list)


if __name__ == "__main__":
    print("gpu is available:")
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reverse_engineer()
