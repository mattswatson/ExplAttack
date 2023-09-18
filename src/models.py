import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=(5, 5), stride=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(5, 5), stride=1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MLP(nn.Module):
    def __init__(self, num_features=784, hidden_1_size=412, hidden_2_size=512, num_classes=10, dropout_prob=0.25):
        super(MLP, self).__init__()
        self.num_features = num_features

        self.fc1 = nn.Linear(self.num_features, hidden_1_size)
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.fc3 = nn.Linear(hidden_2_size, num_classes)
        self.droput = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = x.view(-1, self.num_features)
        x = self.relu(self.fc1(x))

        x = self.droput(x)
        x = self.relu(self.fc2(x))

        x = self.droput(x)

        x = self.fc3(x)

        output = self.softmax(x)
        return output