import sys, os
import math, random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple
import datetime
import pandas as pd

import gymnasium as gym
import gym_race


### DQN Agent #######
class DQN(nn.Module):
    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(observation_space, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.head = nn.Linear(32, action_space)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x)
    

#### Replay Memory ######
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
    

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def full (self):
        return len(self.memory)== self.capacity
#####################################


"""
this imports race_env.py (a gym env) and pyrace_2d.py (the race game) and registers the env as "Pyrace-v1"
"""
"""register(
    id='Pyrace-v1',
    entry_point='gym_race.envs:RaceEnv',
    max_episode_steps=2000,
)"""

VERSION_NAME = 'MODIFIED_ENV_DQN_TARGET_NETWORK' # the name for our model

REPORT_EPISODES  = 20# report (plot) every...
DISPLAY_EPISODES = 10 # display live game every...


############################

def simulate(learning=True): # LEARN
    print('simulating')
    global q_table
    explore_rate  = get_explore_rate(0)
    discount_factor = DISCOUNT_FACTOR
    total_reward = 0
    total_rewards = []
    training_done = False
    threshold = 200
    
    LEARNING = learning

    max_reward = -10_000

    env.set_view(True)

    if training_done:
        return

    for episode in range(NUM_EPISODES):
        if episode > 0:
            total_rewards.append(total_reward)

            if LEARNING and episode % REPORT_EPISODES == 0 and episode >= threshold:
                
                ################## Plot Results with rolling mean ###################
                # Calculate the rolling mean
                rolling_window = 50  # Adjust this for the desired rolling window size
                rolling_mean_rewards = pd.Series(total_rewards).rolling(rolling_window, min_periods=1).mean()

                # Plot the original rewards with crosses as markers
                plt.scatter(range(len(total_rewards)), total_rewards, marker='x', label='Episode Rewards', color='blue')

                # Plot the rolling mean line
                plt.plot(rolling_mean_rewards, label=f'Rolling Mean ({rolling_window} episodes)', color='green')

                plt.ylabel('Rewards')
                plt.show(block=False)
                plt.pause(0.1)

                ######Creating file if one does not exist#######
                if not os.path.exists(VERSION_NAME):
                    os.makedirs(VERSION_NAME)
                    print('Making new directory')
                ################################################


                ############# SAVING DQN WEIGHTS ######################
                file = f'{VERSION_NAME}/weights_{episode}'
                #current_datetime = datetime.datetime.now().strftime('%Y-%m-%d')
                torch.save(policy_net.state_dict(), file)
                print(f'Weights {episode} saved \n')
                #####################################################




        obv, _ = env.reset()
        state_0 = obv
        total_reward = 0
        if not LEARNING:
            env.pyrace.mode = 2 # continuous display of game

        if episode >= threshold:
            explore_rate = 0.002

        for t in range(MAX_T):
            action = select_action(state_0, explore_rate if LEARNING else 0)
            obv, reward, done, _, info = env.step(action)

            ########## Normalizing state definition before putting into memory and DQN
            norm = [200, 200, 200, 200, 200, 10]
            obv = [a / b for a, b in zip(obv, norm)]

            state= obv
            ####################################################
            


            if sum(obv) != sum(state):
                print('WARNING',obv,state)

            env.remember(state_0, action, reward, state, done)

            ######### Saving episode to memory ##########
            memory.push(state_0, action, reward, state, done)
            #############################################

            total_reward += reward
        
            ############# EXPERINECE REPLAY #################
            if memory.full() and LEARNING:
                experience_replay()
            #################################################

            if LEARNING and t % TARGET_UPDATE == 0:
            ######### update target with policy weights ##########
                target_net.load_state_dict(policy_net.state_dict())
            #####################################################

            # Setting up for the next iteration
            state_0 = state
            
            if ((episode % DISPLAY_EPISODES == 0) and episode >= threshold )or (env.pyrace.mode == 2):
                """
                env.render(msgs=['SIMULATE',
                                 f'Episode: {episode}',
                                 f'Time steps: {t}',
                                 f'check: {info["check"]}',
                                 f'dist: {info["dist"]}',
                                 f'crash: {info["crash"]}',
                                 f'Reward: {total_reward:.0f}',
                                 f'Max Reward: {max_reward:.0f}'])
                """
                env.set_msgs(['SIMULATE',
                            f'Episode: {episode}',
                            f'Time steps: {t}',
                            f'check: {info["check"]}',
                            f'dist: {info["dist"]}',
                            f'crash: {info["crash"]}',
                            f'Reward: {total_reward:.0f}',
                            f'Max Reward: {max_reward:.0f}'])
                env.render()
            if done or t >= MAX_T - 1:
                if total_reward > max_reward: max_reward = total_reward
                # print("SIMULATE: Episode %d finished after %i time steps with total reward = %f."
                #      % (episode, t, total_reward))
                break

            early_stopping = check_early_stopping(total_rewards)

        if early_stopping and LEARNING:
            print(f"Early stopping triggered at episode {episode}")

            ############# SAVING DQN WEIGHTS ######################
            if not os.path.exists(VERSION_NAME):
                os.makedirs(VERSION_NAME)

            file = f'{VERSION_NAME}/early_stopping_model_episode_{episode}'

            torch.save(policy_net.state_dict(), file)
            #####################################################
            training_done=True


        # Update parameters
        explore_rate  = get_explore_rate(episode)


def load_and_play(episode, NN = True, early_stopping_flag = None):

    # DIRECT LOADING FROM SAVED DATA...
    ########## Loading Weights ############
    if NN and early_stopping_flag is None:
        # Load the weights from the file into the policy_net
        print(f'loading weights to policy')
        file = f'{VERSION_NAME}/weights_{episode}'
        policy_net.load_state_dict(torch.load(file))
        print('loading successful')
    #######################################

    elif NN and early_stopping_flag is not None:
        policy_net.load_state_dict(torch.load(early_stopping_flag))
        print('loading successful')
    # play game
    simulate(learning=False)


def experience_replay(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    replay = memory.sample(EXP_REP_BATCH_SIZE)

    # Unzip the memory tuples
    states, actions, rewards, next_states, done = zip(*replay)

    states = np.array(states)
    next_states = np.array(next_states)

    # Convert lists to PyTorch tensors and move them to the specified device
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    done = torch.tensor(done, dtype=torch.float32).to(device)  # Convert 'done' to a PyTorch tensor

    # Create a mask for non-terminal states
    non_terminal_mask = 1 - done  # 1 if not done, 0 if done

    # Calculate Q-values for the current state using the policy network
    Q_current = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Calculate the target Q-values using the Bellman equation
    Q_target = rewards + GAMMA * non_terminal_mask * target_net(next_states).max(1).values

    # Compute the loss using Huber loss
    loss = F.smooth_l1_loss(Q_current, Q_target)

    # Optimize the policy network
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-CLIP, CLIP)
    optimizer.step()

    return loss.item()


def select_action(state, explore_rate):
    ############# state to tensor for DQN action #############
    state = torch.tensor(state, dtype=torch.float32)
    ##########################################################
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        ############# Action = index of highest qvalue action #############
        action = policy_net(state).argmax().item()
        ###################################################################
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def DQN_agent():
    ################## Agent #######################
    policy_net = DQN(env.observation_space.shape[0], env.action_space.n)
    print('policy initiated')
    
    # Create target_net for dual DQN
    target_net = DQN(env.observation_space.shape[0], env.action_space.n)
    print('target initiated')

    # Load the weights from policy_net into target_net
    target_net.load_state_dict(policy_net.state_dict())
    print('target weights loaded')

    ########## DEFINING OPTIMIZER #############
    optimizer = optim.Adam(policy_net.parameters(), lr = LEARNING_RATE, weight_decay=1e-5 )

    return policy_net, target_net, optimizer
    ###########################################

############ Early stopping mechanism #########################
def check_early_stopping(total_rewards, threshold=10000):
    if len(total_rewards) >= EARLY_STOPPING_COUNTER:
        last_10_rewards = total_rewards[- EARLY_STOPPING_COUNTER:]
        mean_reward = sum(last_10_rewards) / EARLY_STOPPING_COUNTER
        return mean_reward >= threshold
    return False


if __name__ == "__main__":

    env = gym.make("Pyrace-v1")

    NUM_BUCKETS  = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_ACTIONS  = env.action_space.n
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
    print(NUM_BUCKETS,NUM_ACTIONS,STATE_BOUNDS)
    """
    (11, 11, 11, 11, 11) 
    3 
    [(0, 10), (0, 10), (0, 10), (0, 10), (0, 10)]
    """
    MIN_EXPLORE_RATE  = 0.001
    MIN_LEARNING_RATE = 0.2
    DISCOUNT_FACTOR   = 0.99

    DECAY_FACTOR = np.prod(NUM_BUCKETS, dtype=float) / 10.0
    print(DECAY_FACTOR)
    """
    16105.1
    """
    NUM_EPISODES = 65_000
    MAX_T = 3000
    #MAX_T = np.prod(NUM_BUCKETS, dtype=int) * 100

    ###### DQN Variables #######
    CLIP = 0.5
    EXP_REP_BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    GAMMA = 0.95

    
    MEMORY_SIZE = 100_00
    TARGET_UPDATE = 600_0
    EARLY_STOPPING_COUNTER = 10

    print(f'Observation Space: {env.observation_space}')
    print(f'Action Space : {env.action_space}')

    #################DQN_agent#########################
    policy_net, target_net, optimizer = DQN_agent()
    ###################################################

    ################# Experinece Replay ###############
    memory = ReplayMemory(MEMORY_SIZE)
    ###################################################
    #-------------
    # simulate() # LEARN

    load_and_play(200)
    # load_and_play(35000)
    #-------------
