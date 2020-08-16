import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random

class DQNAgent():
    def __init__(self):
        self.nb_actions = nb_actions
        self.model=model
        self.memory=memory
        self.target_model_update=target_model_update
        self.nb_steps_warmup=nb_steps_warmup


class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.index = 0
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)

    def push(self,experience):
        if len(self.buffer) > self.buffer_size:

        else:
            self.buffer.append(experience)
        self.index = (self.index + 1) % self.buffer_size
    def sample(self):
        indices = np.random.choice(len(self.buffer), batch_size)
        obs, action, reward, next_obs, done = zip(*[self.buffer[i] for i in indices])
        return (torch.as_tensor(obs),
                torch.as_tensor(action), 
                torch.as_tensor(reward, dtype=torch.float32),
                torch.as_tensor(next_obs), 
                torch.as_tensor(done, dtype=torch.uint8),
                torch.as_tensor(weights, dtype=torch.float32))


class Model(nn.Module,nb_actions):
    def __init__(self):
        super(Model,self).__init__()
        self.fc1 = nn.Linear(nb_actions,256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64,nb_actions)
    
    def forward(self,x):
        x = nn.Flatten(x)
        x = F.relu(self.fc1(x))
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
