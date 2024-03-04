import sys, os
import math, random
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import gym_race
"""
this imports race_env.py (a gym env) and pyrace_2d.py (the race game) and registers the env as "Pyrace-v1"

register(
    id='Pyrace-v1',
    entry_point='gym_race.envs:RaceEnv',
    max_episode_steps=2000,
)
"""
VERSION_NAME = 'QT_v02' # the name for our model

REPORT_EPISODES  = 500 # report (plot) every...
DISPLAY_EPISODES = 100 # display live game every...

def simulate(learning=True): # LEARN
    global q_table
    learning_rate = get_learning_rate(0)
    explore_rate  = get_explore_rate(0)
    discount_factor = DISCOUNT_FACTOR
    total_reward = 0
    total_rewards = []
    training_done = False
    threshold = 1000
    
    LEARNING = learning

    max_reward = -10_000

    env.set_view(True)

    for episode in range(NUM_EPISODES):

        if episode > 0:
            total_rewards.append(total_reward)

            if LEARNING and episode % REPORT_EPISODES == 0:
                plt.plot(total_rewards)
                plt.ylabel('rewards')
                # plt.show()
                plt.show(block=False)
                plt.pause(.1)
                file = f'models_{VERSION_NAME}/memory_{episode}'
                env.save_memory(file)
                file = f'models_{VERSION_NAME}/q_table_{episode}'
                # print(q_table) # homogeneus types
                data = q_table
                if data.shape[0] == 11: # q_table
                    print('max min',data.max(),data.min(), 'total', data.sum())
                    print('zeros', np.count_nonzero(data == 0), 'total', data.size)

                np.save(file,q_table)
                print(file,'saved')

        obv, _ = env.reset()
        state_0 = state_to_bucket(obv)
        total_reward = 0
        if not LEARNING:
            env.pyrace.mode = 2 # continuous display of game

        if episode >= threshold:
            explore_rate = 0.01

        for t in range(MAX_T):
            action = select_action(state_0, explore_rate if LEARNING else 0)
            obv, reward, done, _, info = env.step(action)
            state = state_to_bucket(obv)
            if sum(obv) != sum(state):
                print('WARNING',obv,state)
            env.remember(state_0, action, reward, state, done)
            total_reward += reward
            
            if LEARNING:
                # Update the Q based on the result
                best_q = np.amax(q_table[state])
                q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state
            
            if (episode % DISPLAY_EPISODES == 0) or (env.pyrace.mode == 2):
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
        # Update parameters
        explore_rate  = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)

def load_and_play(episode):
    global q_table
    """
    # RECONSTRUCT QTABLE FROM MEMORY - NOT USED...
    print("Start loading history")
    history_list = [f'models_{VERSION_NAME}/memory_{episode}'+'.npy']

    # load data from history file
    print("Start updating q_table")
    discount_factor = DISCOUNT_FACTOR
    for list in history_list:
        history = load_data(list)
        learning_rate = get_learning_rate(0)
        print(list)
        file_size = len(history)
        print("file size : " + str(file_size),history.shape)
        i = 0
        for data in history:
            state_0, action, reward, state, done = data
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])
            if done == True:
                i += 1
                learning_rate = get_learning_rate(i)

    print("Updating q_table is complete")
    """
    # DIRECT LOADING FROM SAVED DATA...
    print("Start loading q_table")
    file = f'models_{VERSION_NAME}/q_table_{episode}'+'.npy'
    q_table = load_data(file)
    # print(q_table)
    file = f'models_{VERSION_NAME}/memory_{episode}'+'.npy'
    memory = load_data(file)
    i = np.count_nonzero(memory[:,4] == True) # episodes

    discount_factor = DISCOUNT_FACTOR
    learning_rate = get_learning_rate(i)

    # play game
    simulate(learning=False)


def select_action(state, explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        action = int(np.argmax(q_table[state]))
    return action

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

def load_data(file):
    data = np.load(file,allow_pickle=True)
    print(type(data))
    print(data.shape)
    # print(data[-1])
    if data.shape[0] == 11: # q_table
        print('max min',data.max(),data.min(), 'total', data.sum())
        print('zeros', np.count_nonzero(data == 0), 'total', data.size)
    else: # memory
        print('episodes', np.count_nonzero(data[:,4] == True))
    return data


if __name__ == "__main__":

    env = gym.make("Pyrace-v1")
    if not os.path.exists(f'models_{VERSION_NAME}'): os.makedirs(f'models_{VERSION_NAME}')

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
    MAX_T = 2000
    #MAX_T = np.prod(NUM_BUCKETS, dtype=int) * 100

    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)
    print(q_table.shape)
    #print(q_table)

    #-------------
    #simulate() # LEARN
    load_and_play(5000)
    # load_and_play(35000)
    #-------------
