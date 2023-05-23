import FLAME
import torch
import random
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch
from sklearn import svm


class model_train:
    def __init__(self, normal_models, danger_models):
        self.modelx = []
        self.modely = []
        for i in range(len(normal_models)):
            mo = FLAME.torchfile2np(normal_models[i])[0:10739712]
            #mo = np.hsplit(mo, 4096)
            #x = np.mean(mo, axis=1)
            #mo = (x - np.min(x)) / (np.max(x) - np.min(x))
            self.modelx.append(mo)
            self.modely.append(0)

        for i in range(len(danger_models)):
            mo = FLAME.torchfile2np(danger_models[i])[0:10739712]
            #mo = np.hsplit(mo, 4096)
            #x = np.mean(mo, axis=1)
            #mo = (x - np.min(x)) / (np.max(x) - np.min(x))
            self.modelx.append(mo)
            self.modely.append(1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.modelx, self.modely, test_size=0.2,
                                                                                random_state=random.randint(1, 100))

        self.X_train = torch.tensor(np.array(self.X_train))
        self.y_train = torch.squeeze(torch.tensor(self.y_train))
        self.X_test = torch.tensor(np.array(self.X_test))
        self.y_test = torch.squeeze(torch.tensor(self.y_test))
        print(self.X_train.shape, self.y_train.shape)

    def train_torch(self, learning_rate=0.02, epochs=100, log_interval=10):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(4096, 512)
                self.fc2 = nn.Linear(512, 64)
                self.fc3 = nn.Linear(64, 2)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return F.log_softmax(x, -1)

        net = Net()
        print(net)

        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.NLLLoss()

        for epoch in range(epochs):
            for i in range(len(self.X_train)):
                optimizer.zero_grad()
                input = self.X_train[i].to(torch.float32)
                net_out = net(input)
                loss = criterion(net_out, self.y_train[i])
                loss.backward()
                optimizer.step()
                if i % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, i, len(self.X_train),
                        100. * i / len(self.X_train), loss.data))

        test_loss = 0
        correct = 0
        for i in range(len(self.X_test)):
            input = self.X_test[i].to(torch.float32)
            net_out = net(input)
            print(net_out)
            test_loss += criterion(net_out, self.y_test[i]).data
            pred = net_out.data.max(0)[1]  # get the index of the max log-probability
            print(pred)
            print(self.y_test)
            correct += pred.eq(self.y_test[i]).sum()

        test_loss /= len(self.X_test)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.X_test),
            100. * correct / len(self.X_test)))
        self.net = net

    def predicted_torch(self, unmodel):
        mo = FLAME.torchfile2np(unmodel)[0:10739712]
        mo = np.hsplit(mo, 4096)
        x = np.mean(mo, axis=1)
        mo = (x - np.min(x)) / (np.max(x) - np.min(x))
        mo = torch.tensor(np.array(mo))
        input = mo.to(torch.float32)
        net_out = self.net(input)
        pred = net_out.data.max(0)[1]
        return pred

    def train(self):
        self.svm = svm.SVC(C=0.5, kernel='linear', decision_function_shape='ovr')
        self.svm.fit(self.modelx, self.modely)

    def predicted(self, unmodel):
        t = FLAME.torchfile2np(unmodel)[0:10739712]
        predict = self.svm.predict([t])
        return predict
