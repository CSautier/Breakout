import os
import gym
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PPO training')
parser.add_argument('--load', default=False, help='Whether or not to load pretrained weights. You must have started an alread trained net for it to work',
                    dest='load', type=str2bool)
parser.add_argument('--render', default=1, help='How many windows you want to see. This slows the training a bit',
                    dest='render', type=int)
parser.add_argument('--processes', default=20, help='Number of processes that plays the game. Note: there will always be a process to learn from it',
                    dest='processes', type=int)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
GAMMA=0.95
BATCH_SIZE=400
LOSS_CLIPPING=0.1
ENTROPY_LOSS=1e-4
env = gym.make('Breakout-v0')

def conv2d_size_out(shape, filters, kernel_size , stride = 1, padding=0):
    if(len(shape)>3):
        raise Exception('Expected dimemsion 3')
    out_shape=(filters,)
    if isinstance(kernel_size, int):
        kernel_size=(kernel_size,kernel_size)
    if isinstance(stride, int):
        stride=(stride,stride)
    if isinstance(padding, int):
        padding=(padding,padding)
    out_shape+= (((shape[1] + 2*padding[0]) - (kernel_size[0] - 1) - 1) // stride[0] + 1,)
    out_shape+= (((shape[2] + 2*padding[1]) - (kernel_size[1] - 1) - 1) // stride[1] + 1,)
    return out_shape

def preprocess_state(observation):
    return np.ascontiguousarray(observation.transpose((2, 0, 1)), dtype=np.float32) / 255

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype='float32')[y]).to(device)

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 49 * 37, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fcactor = nn.Linear(84, 4)
        self.fccritic = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 49 * 37)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actor = F.softmax(self.fcactor(x), dim=1)
        if self.training:
            critic = self.fccritic(x)
            return actor, critic
        return actor

def cpu_thread(render, memory_queue, process_queue, common_dict):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        pid = os.getpid()
        print('process started with pid: {}'.format(pid), flush=True)
        while True:
            observation = env.reset()
            done=False
            reward_list=[]
            action_list=[]
            observation_list=[]
            while not done:
                observation_list.append(observation)
                process_queue.put((pid, observation))
                while pid not in common_dict:
                    time.sleep(0.01)
                predicted = common_dict[pid]
                del common_dict[pid]
                action = Categorical(predicted).sample().item()
                observation,reward,done,info = env.step(action)
                action_list.append(action)
                reward_list.append(reward*0.1)
                if render:
                    env.render()
            print(predicted)
            print('Score: {}'.format(sum(reward_list)), flush=True)
            for i in range(len(reward_list)-2, -1, -1):
                reward_list[i]+=reward_list[i+1] * GAMMA #compute the discounted obtained reward for each step
            for i in range(len(observation_list)):
                memory_queue.put((observation_list.pop(), reward_list.pop(), action_list.pop()))
    except Exception as e: print(e, flush=True)


def gpu_thread(load, memory_queue, process_queue, common_dict):
    #the only thread that has an access to the gpu, it willl then perform all the NN computation
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        print('process started with pid: {}'.format(os.getpid()), flush=True)
        ppo = PPO()
        ppo.to(device)
        optimizer = optim.Adam(ppo.parameters())
        if load:
            checkpoint=torch.load('./model/breakout.pt')
            ppo.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ppo.eval()
        while True:
            if memory_queue.qsize()>=BATCH_SIZE:
                ppo.train()
                states=[]
                rewards=[]
                actions=[]
                for j in range(BATCH_SIZE):
                    temp = memory_queue.get(True)
                    states.append(preprocess_state(temp[0]))
                    rewards.append(temp[1])
                    actions.append(temp[2])
                states=np.array(states)
                prob, reward_pred=ppo.forward(torch.from_numpy(states).to(device))
                lossentropy = - ENTROPY_LOSS * torch.mean(Categorical(prob).entropy())
                prob= (prob*to_categorical(actions, 4)).sum(dim=-1)
                old_prob=prob.detach()
                r=(prob+1e-10)/(old_prob+1e-10)
                rewards = torch.from_numpy(np.array(rewards, dtype='float32')).to(device)
                advantage = rewards - reward_pred.squeeze().detach()
                lossactor= -1*torch.mean(torch.min(r * advantage, torch.clamp(r, min = (1.-LOSS_CLIPPING), max = (1.+LOSS_CLIPPING)) * advantage))
                losscritic = F.smooth_l1_loss(reward_pred, rewards.unsqueeze(1))
                print('Loss actor: {0:7.3f}\tLoss critic: {1:7.3f}\tLoss entropy: {2:7.3f}'.format(100*lossactor, 100*losscritic, 100*lossentropy), flush=True)
                optimizer.zero_grad()
                (lossactor + losscritic + lossentropy).backward()
                optimizer.step()
                torch.save({
                    'model_state_dict': ppo.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, './model/breakout.pt')
            else:
                ppo.eval()
                pid, observation = process_queue.get(True)
                with torch.no_grad():
                    #I know that I preprocess each frame twice, but this is much more RAM effective to store the image as an array of uint8 than float32
                    common_dict[pid]= ppo.forward(torch.from_numpy(preprocess_state(observation)).unsqueeze(0).to(device))[0].to("cpu")
    except Exception as e: 
        print(e)
        print('saving before interruption', flush=True)
        torch.save({
                'model_state_dict': ppo.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, './model/breakout.pt')

def main(args):
    #create shared variables between all the processes
    manager = mp.Manager()
    #used to send the results of the net
    common_dict = manager.dict()
    #a queue of batches to be fed to the training net
    mem_queue = manager.Queue(800)
    #a queue of operations pending
    process_queue = manager.Queue(args.processes)
    
    #initializes all workers
    pool = mp.Pool(args.processes+1)
    try:
        pool.apply_async(gpu_thread, (True, mem_queue, process_queue, common_dict))
        for i in range(min(args.render, args.processes)):
            pool.apply_async(cpu_thread, (True, mem_queue, process_queue, common_dict))
        for i in range(args.processes-args.render):
            #starts the player
            pool.apply_async(cpu_thread, (False, mem_queue, process_queue, common_dict))

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.join()
    
if __name__ == "__main__":
    args = parser.parse_args()
    try:
        mp.set_start_method('spawn')
    except: pass
    main(args)
