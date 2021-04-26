# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import axes3d
from tensorflow import nn ## de goi cac active function
from sklearn import tree
from mpl_toolkits.mplot3d import axes3d
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Input,Activation, Dropout, Flatten, Dense
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from collections import namedtuple
import matplotlib
from keras.callbacks import TensorBoard
from keras.models import load_model
from IPython.display import clear_output
from keras.preprocessing import image
from keras.utils import np_utils ## dung de categorical cac label
from sklearn.model_selection import train_test_split  ## dung de tach bo test ra
from sklearn.datasets import load_iris
import time
import datetime
from math import pow
import random
from keras.callbacks import ReduceLROnPlateau
from collections import deque
import csv
from keras.layers import PReLU
import math
from keras.callbacks import ModelCheckpoint, EarlyStopping
import sys
import os
import pickle
from keras.layers.normalization import BatchNormalization
import h5py
import simplejson as json
from tqdm import tqdm
from keras_radam import RAdam
from sklearn import preprocessing
from IPython.display import clear_output
from keras.utils import  to_categorical
from keras.callbacks import ModelCheckpoint
from itertools import count
from PIL import Image
import PIL
import math
from keras.callbacks import History
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
import glob
from keras.layers import LeakyReLU
from mpi_adam import MpiAdam
import tf_util as U
from agent import Agent
from multiprocessing import Process
from Environment import *
TRAIN_ITERATIONS = 50
MAX_EPISODE_LENGTH = 128
TRAJECtoRY_BUFFER_SIZE = 32
BATCH_SIZE = 16
RENDER_EVERY = 100
AGGREGATE_STATS_EVERY = 1
MODEL_NAME = "model"
def train():
    os.chdir("D:/Study/RL/RLAChips-V11")
    img_height, img_width = maze.shape
    env = Qmaze(maze)
    env.get_list_centroid()
    num_states = maze.size
    state_dim = env.state_dim
    input_dim, output_dim = state_dim, num_actions
    lr, gamma, loss_clipping, c1, lamda = 1e-6, 0.90 , 0.2, 0.001, 0.95
    agent = Agent(input_dim, output_dim, lr, gamma, loss_clipping, c1,lamda)

    tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
    EPISODES = 200000
    AGGREGATE_STATS_EVERY = 1
    ep_rewards=[]
    ep_loss = []
    ep_pw = []
    win_count = 0
    num_non_cell = 32
    supervision_factor = 0.3
    for e in range(1,EPISODES+1):
        rat_cell = random.choice(env.free_cells)
        non_available_cell = random.choices(env.free_cells,k=num_non_cell)
        target_cell = random.choice(env.free_target_cells)
        #target_cell = (34,33)
        state = env.reset(rat_cell,target_cell,non_available_cell)
        r_sum = 0
        loss_sum =0
        reward_sum = 0
        power_sum = 0
        done = False
        if e % 1000==0:
            clear_output(wait=True)
        for cnt_step in range(MAX_EPISODE_LENGTH):
            #show(env)
            #clear_output(wait=True)
            plt.close()
            state = np.reshape(state,(-1,state_dim))
            action,pi_vec =  agent.act(state)
            ran_supervised = np.random.uniform(0,1)
            if ran_supervised < supervision_factor:
                env.supervised = 1
                action = env.valid_actions()[0]
                env.supervised = 0
            #get s_,r,done
            env.current_action = action
            next_state, reward, done= env.act(action)
            next_state = np.reshape(next_state,(-1,state_dim))
#             if cnt_step%10==0:
#                 env.non_available_cell = random.choices(env.free_cells,k=num_non_cell)
            reward_sum += reward
            if done == 'not_over':
                done = False
            else :
                done = True
            mask = not done
            agent.remember(state, action,mask,pi_vec, reward)
            state = next_state
            if done == True :
                rat_cell = random.choice(env.free_cells)
                non_available_cell = random.choices(env.free_cells,k=num_non_cell)
                target_cell = random.choice(env.free_target_cells)
                #target_cell = (34,33)
                state = env.reset(rat_cell,target_cell,non_available_cell)
                loss_sum +=10*np.log10(3/4)
            if reward==2 :
                loss = 0
                win_count+=1
            elif reward==0 :
                loss = 0
            elif reward == -1 :
                loss = 0
            elif reward == -1.5 :
                try :
                    loss = df["{}_to_{}".format(action_node_dict[env.old_action],action_node_dict[env.current_action])]
                except :
                    loss = 0
            else :
                loss = reward
            loss_sum+=loss*10
            try :
                power_sum += pc["{}_to_{}".format(action_node_dict[env.old_action],action_node_dict[env.current_action])]
            except :
                power_sum +=0
            env.old_action = action
            print("Episode : ",e, " Probability : ",pi_vec,"\n Action : ",action," Reward : ",reward," Loss : ",loss*10," Win count : ",win_count)
        agent.train_models()
        ep_rewards.append(reward_sum)
        ep_loss.append(loss_sum)
        ep_pw.append(power_sum)
        if  e%AGGREGATE_STATS_EVERY==0 or e == 1 :
            print("updating stats...")
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            average_loss = sum(ep_loss[-AGGREGATE_STATS_EVERY:]) / len(ep_loss[-AGGREGATE_STATS_EVERY:])
            average_pwc = sum(ep_pw[-AGGREGATE_STATS_EVERY:]) / len(ep_pw[-AGGREGATE_STATS_EVERY:])
            #tensorboard.update_stats(tranmission_loss=loss_sum)
            tensorboard.step = e
            tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward,
                                           reward_max=max_reward,avarage_tranmission_loss = average_loss,
                                           avarage_power_consumption = average_pwc,
                                           epsilon=1)
            agent.save_model()

class MultiProcess(object):
    def __init__(self, num_process=4):
        self.num_process = num_process
        assert self.num_process > 0
        pass

    def __call__(self):
        process_list = []
        for _ in range(self.num_process):
            p = Process(target=train)
            p.start()
            process_list.append(p)

        for _ in range(len(process_list)):
            p = process_list[_]
            p.join()

        pass
if __name__ == "__main__":
    tstart = time.time()
    multi = MultiProcess(2)
    multi()
    tend = time.time()