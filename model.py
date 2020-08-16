import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random

class Model(nn.Module,n_action):
    def __init__(self):
        super(Model,self).__init__()
        self.fc1 = nn.Linear(n_action,256)
        self.fc2 = nn.Linear(256,n_action)
    
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    # epsilon-greedy. 確率epsilonでランダムに行動し, それ以外はニューラルネットワークの予測結果に基づいてgreedyに行動します. 
    def act(self, obs, epsilon):
        if random.random() < epsilon:
            action = random.randrange(self.n_action)
        else:
            # 行動を選択する時には勾配を追跡する必要がない
            with torch.no_grad():
                action = torch.argmax(self.forward(obs.unsqueeze(0))).item()
        return action
