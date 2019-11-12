
from IPython.display import clear_output
import matplotlib.pyplot as plt
import gym_super_mario_bros
import logging
import gym
import numpy as np
import sys
from nes_py.wrappers import JoypadSpace
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from collections import deque
import cv2

cv2.ocl.setUseOpenCL(False)


movements = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up'],
]

## get_ipython().run_line_magic('matplotlib', 'inline')

# gpu 사용 여부
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)

# print(torch.cuda.get_device_name(0))

# random 움직임
'''
done = True
for step in range(1000):
    if done:
        state = env.reset()
    action = env.action_space.sample()
    #print(action)
    state, reward, done, info = env.step(action)
    env.render()
env.close()
'''


# 학습 진행 결과
def plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


# 논문에서 사용된 84 * 84 GREY SCALE (일단 미사용)
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, self.height, self.width), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame[:,:,None]
        return np.swapaxes(frame,2,0)


# image -> gym space 0.0 ~ 1.0 , n * w * rgb -> channel * n * w
class ImageToPytorch(gym.ObservationWrapper):
    def __init__(self, env):
        #print(10)
        super(ImageToPytorch, self).__init__(env)
        old_shape = self.observation_space.shape
        # print(old_shape)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,  # 0 ~ 1 값으로 바꿈)
                                                shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.uint8)
        # print(self.observation_space.shape)

    def observation(self, observation):
        #print(11)
        return np.swapaxes(observation, 2, 0)


# (s, a, r) Tuple 집합 생성 replay initial 부터 쌓기 시작 limit size 유지
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)
    def get_list(self,i):
        oshape = self.buffer[i][0].shape
        for i in range(oshape[2]):
            for j in range(oshape[3]):
                f.write("%d " % self.buffer[i][0][0][0][i][j])
        f.write('\n')
        f.write("%d \n" % self.buffer[i][1])
        f.write("%d \n" % self.buffer[i][2])
        nshape = self.buffer[i][3].shape
        for i in range(nshape[2]):
            for j in range(nshape[3]):
                f.write("%d " % self.buffer[i][3][0][0][i][j])
        f.write('\n')
        if self.buffer[i][4] == True:
            f.write('1')
        else:
            f.write('0')
        f.write('\n')


        # epsilon greedy 선택을 위한 parameter 초기화
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 300000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
    -1. * frame_idx / epsilon_decay)


# 모델 정의
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )  # feature selection
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # reshape
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):  # epsilon-greedy selection
        if random.random() > epsilon:
            with torch.no_grad():
                state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0))
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0].item()
            # print("chosen", action)
        else:
            action = env.action_space.sample()
            # print("random", action)
        return action

_env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(_env, movements)
def MyWrapper(env):
    return WarpFrame(env)

env = MyWrapper(env)

#print(env.observation_space.shape)

model = DQN(env.observation_space.shape, env.action_space.n)

if USE_CUDA:
    model = model.cuda()
    print("cuda : O")
else:
    print("cuda : X")

optimizer = optim.Adam(model.parameters(), lr=0.00001)

replay_initial = 10000
replay_buffer = ReplayBuffer(500000)

num_frames = 300000
batch_size = 32
gamma = 0.99

losses = []
all_rewards = []
episode_reward = 0

def compute_td_loss(batch_size):  # q 함수 정의 및 loss 구하는 함수
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    with torch.no_grad():
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))  # no grad tensor 정의
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = model(state)
    next_q_values = model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)  # 논문에 정의된 식

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()  # 변화율 0으로
    loss.backward()  # backprop
    optimizer.step()

    return loss

model.load_state_dict(torch.load('saved_model_state.pt'))
model.eval()

with open("losses.txt", "r") as file1:
    f_list = [float(i) for line in file1 for i in line.split(' ') if i.strip()]
    for i in range(len(f_list)):
        losses.append(float(f_list[i]))

with open("rewards.txt","r") as file2:
    f_list = [float(i) for line in file2 for i in line.split(' ') if i.strip()]
    for i in range(len(f_list)):
        all_rewards.append(float(f_list[i]))

with open("replay.txt","r") as file3:
    line = file3.readline()
    replay_len = int(line)
    for i in range(replay_len):
        temp_state = np.zeros((1, 84, 84))
        temp_nextstate = np.zeros((1,84,84))
        line = file3.readline()
        array = []
        if line.strip():
            for x in line.split(' '):
                if x == '\n':
                    break
                array.append(int(x))
        k = 0
        for x in range(84):
            for y in range(84):
                temp_state[0][x][y] = array[k]
                k = k + 1
        temp_action = int(file3.readline())
        temp_reward = int(file3.readline())
        line = file3.readline()
        array = []
        if line.strip():
            for x in line.split(' '):
                if x == '\n':
                    break
                array.append(int(x))
        k = 0
        for x in range(84):
            for y in range(84):
                temp_nextstate[0][x][y] = array[k]
                k = k + 1
        dflag = int(file3.readline())
        if dflag == 1:
            temp_done = True
        else:
            temp_done = False
        replay_buffer.push(temp_state,temp_action,temp_reward,temp_nextstate,temp_done)

print('Load Complete')

trainF = True
# trainF = False

if trainF == False:
    state = env.reset()
    num_frames = 15000
    for frame_idx in range(1, num_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        action = model.act(state, epsilon)
        # print(action)
        next_state, reward, done, _ = env.step(action)
        # print(reward)
        env.render()
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > replay_initial:
            loss = compute_td_loss(batch_size)
            losses.append(loss.item())

        if frame_idx % 1000 == 0:
            print('frame : %d' % (frame_idx), ' ', all_rewards)
            # plot(frame_idx, all_rewards, losses)

    torch.save(model.state_dict(), 'saved_model_state.pt')
    f = open('replay.txt', 'w')
    f.write("%d" % replay_buffer.__len__())
    f.write('\n')
    for ho in range(replay_buffer.__len__()):
        replay_buffer.get_list(ho)

    with open('losses.txt','w') as lf:
        for item in losses:
            lf.write('%f ' % item)

    with open('rewards.txt','w') as rf:
        for item in all_rewards:
            rf.write('%d ' % item)
    plt.plot(losses)

# play
'''
Max_Time = 35000
Frame_Count = 0
while Frame_Count < Max_Time:
    # choose action 모델로만
    epsilon = 0
    action = model.act(state, epsilon)
    state, reward, done, info = env.step(action)
    if done:
        print(reward)
        break
    env.render()
    Frame_Count = Frame_Count + 1
'''