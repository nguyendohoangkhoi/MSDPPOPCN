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
maze=[]
def get_map():
    maze=[]
#     for i in range(30):
#         if (i==0 or i==5 or i==6 or i==11 or i==12 or i==17 or i==18 or i == 23 or i== 24 or i==29 or i == 30):
#               maze.append([0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0])
#         elif (i==2 or i==8 or i==14 or i == 20 or i==26 ) :
#               maze.append([1,0.9,0.95,0.9,0.9,1,1,0.9,0.95,0.9,0.9,1,1,0.9,0.95,0.9,0.9,1,1,0.9,0.95,0.9,0.9,1,1,0.9,0.95,0.9,0.9,1])
#         else :
#               maze.append([1,0.9,0.9,0.9,0.9,1,1,0.9,0.9,0.9,0.9,1,1,0.9,0.9,0.9,0.9,1,1,0.9,0.9,0.9,0.9,1,1,0.9,0.9,0.9,0.9,1])
#     maze = np.asarray(maze)
    for i in range(35):
        if (i==0 or i==4 or i==5 or i==9 or i==10 or i==14 or i==15 or i==19 or i==20 or i==24 or i==25 or i==29 or i==30 or i==34 or i==35):
            maze.append([0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0])
        elif (i==2 or i==7 or i==12 or i==17 or i==22 or i==27 or i==32) :
            maze.append([1,0.9,0.95,0.9,1,1,0.9,0.95,0.9,1,1,0.9,0.95,0.9,1,1,0.9,0.95,0.9,1,1,0.9,0.95,0.9,1,1,0.9,0.95,0.9,1,1,0.9,0.95,0.9,1])
        else :
            maze.append([1,0.9,0.9,0.9,1,1,0.9,0.9,0.9,1,1,0.9,0.9,0.9,1,1,0.9,0.9,0.9,1,1,0.9,0.9,0.9,1,1,0.9,0.9,0.9,1,1,0.9,0.9,0.9,1])
    maze = np.asarray(maze)
    return maze
maze = get_map()
maze = maze.astype(float)
maze.shape
# for i in range(35):
#     if (i==0 or i==4 or i==5 or i==9 or i==10 or i==14 or i==15 or i==19 or i==20 or i==24 or i==25 or i==29 or i==30 or i==34 or i==35):
#         maze.append([0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0])
#     elif (i==2 or i==7 or i==12 or i==17 or i==22 or i==27 or i==32) :
#         maze.append([1,0.9,0.95,0.9,1,1,0.9,0.95,0.9,1,1,0.9,0.95,0.9,1,1,0.9,0.95,0.9,1,1,0.9,0.95,0.9,1,1,0.9,0.95,0.9,1,1,0.9,0.95,0.9,1])
#     else :
#         maze.append([1,0.9,0.9,0.9,1,1,0.9,0.9,0.9,1,1,0.9,0.9,0.9,1,1,0.9,0.9,0.9,1,1,0.9,0.9,0.9,1,1,0.9,0.9,0.9,1,1,0.9,0.9,0.9,1])
visited_mark = 0.8  # Cells visited by the rat will be painted by gray 0.8
rat_mark = 0.5      # The current rat cell will be painteg by gray 0.5
target_mark = 0.7
LEFT1 = 0
LEFT2 = 1
LEFT3 = 2
UP1 = 3
UP2 = 4
UP3 = 5
RIGHT1 = 6
RIGHT2 = 7
RIGHT3 = 8
DOWN1 = 9
DOWN2 = 10
DOWN3 = 11

# Actions dictionary
actions_dict = {
    LEFT1: 'left1',
    LEFT2: 'left2',
    LEFT3: 'left3',
    UP1:'up1',
    UP2:'up2',
    UP3:'up3',
    RIGHT1: 'right1',
    RIGHT2: 'right2',
    RIGHT3: 'right3',
    DOWN1: 'down1',
    DOWN2: 'down2',
    DOWN3: 'down3',
}
state_centroid_dict = {}
num_actions = len(actions_dict)
MODEL_NAME = "model"
# Exploration factor
AGGREGATE_STATS_EVERY = 5
os.chdir("D:/Study/RL/RLAChips-V11/data")
source_data = glob.glob("*.xlsx")
df = {}
pc={}
for file in source_data :
  data = pd.read_excel(file,header=0)
  print(data)
  row = data.loc[data['Lamda']==1.5500].values[0][1:]
  file_name = file.split('.')[0]
  output_loss = row[0]
  power_cons = row[3]
  #print(output_loss)
  df[file_name] = np.log10(output_loss*4/3)
  pc[file_name] = power_cons


###########################################################
df["K1_to_O1"]=df["O1_to_K1"]=df["J1_to_I3"]=df["I3_to_J1"]
df["K1_to_O2"]=df["O1_to_K2"]=df["J1_to_I2"]=df["I3_to_J2"]
df["K1_to_O3"]=df["O1_to_K3"]=df["J1_to_I1"]=df["I3_to_J3"]
df["K1_to_J1"]=df["O1_to_I3"]=df["J1_to_O1"]=df["I3_to_K1"]
df["K1_to_J2"]=df["O1_to_I2"]=df["J1_to_O2"]=df["I3_to_K2"]
df["K1_to_J3"]=df["O1_to_I1"]=df["J1_to_O3"]=df["I3_to_K3"]
df["K1_to_I3"]=df["O1_to_J1"]=df["J1_to_K1"]=df["I3_to_O1"]
df["K1_to_I2"]=df["O1_to_J2"]=df["J1_to_K2"]=df["I3_to_O2"]
df["K1_to_I1"]=df["O1_to_J3"]=df["J1_to_K3"]=df["I3_to_O3"]
###########################################################
df["K3_to_O1"]=df["O3_to_K1"]=df["J3_to_I3"]=df["I1_to_J1"]
df["K3_to_O2"]=df["O3_to_K2"]=df["J3_to_I2"]=df["I1_to_J2"]
df["K3_to_O3"]=df["O3_to_K3"]=df["J3_to_I1"]=df["I1_to_J3"]
df["K3_to_J1"]=df["O3_to_I3"]=df["J3_to_O1"]=df["I1_to_K1"]
df["K3_to_J2"]=df["O3_to_I2"]=df["J3_to_O2"]=df["I1_to_K2"]
df["K3_to_J3"]=df["O3_to_I1"]=df["J3_to_O3"]=df["I1_to_K3"]
df["K3_to_I3"]=df["O3_to_J1"]=df["J3_to_K1"]=df["I1_to_O1"]
df["K3_to_I2"]=df["O3_to_J2"]=df["J3_to_K2"]=df["I1_to_O2"]
df["K3_to_I1"]=df["O3_to_J3"]=df["J3_to_K3"]=df["I1_to_O3"]
###########################################################
df["K2_to_O1"]=df["O2_to_K1"]=df["J2_to_I3"]=df["I2_to_J1"]
df["K2_to_O2"]=df["O2_to_K2"]=df["J2_to_I2"]=df["I2_to_J2"]
df["K2_to_O3"]=df["O2_to_K3"]=df["J2_to_I1"]=df["I2_to_J3"]
df["K2_to_J1"]=df["O2_to_I3"]=df["J2_to_O1"]=df["I2_to_K1"]
df["K2_to_J2"]=df["O2_to_I2"]=df["J2_to_O2"]=df["I2_to_K2"]
df["K2_to_J3"]=df["O2_to_I1"]=df["J2_to_O3"]=df["I2_to_K3"]
df["K2_to_I3"]=df["O2_to_J1"]=df["J2_to_K1"]=df["I2_to_O1"]
df["K2_to_I2"]=df["O2_to_J2"]=df["J2_to_K2"]=df["I2_to_O2"]
df["K2_to_I1"]=df["O2_to_J3"]=df["J2_to_K3"]=df["I2_to_O3"]
###########################################################
###########################################################
pc["K1_to_O1"] = pc["O1_to_K1"] = pc["J1_to_I3"] = pc["I3_to_J1"]
pc["K1_to_O2"] = pc["O1_to_K2"] = pc["J1_to_I2"] = pc["I3_to_J2"]
pc["K1_to_O3"] = pc["O1_to_K3"] = pc["J1_to_I1"] = pc["I3_to_J3"]
pc["K1_to_J1"] = pc["O1_to_I3"] = pc["J1_to_O1"] = pc["I3_to_K1"]
pc["K1_to_J2"] = pc["O1_to_I2"] = pc["J1_to_O2"] = pc["I3_to_K2"]
pc["K1_to_J3"] = pc["O1_to_I1"] = pc["J1_to_O3"] = pc["I3_to_K3"]
pc["K1_to_I3"] = pc["O1_to_J1"] = pc["J1_to_K1"] = pc["I3_to_O1"]
pc["K1_to_I2"] = pc["O1_to_J2"] = pc["J1_to_K2"] = pc["I3_to_O2"]
pc["K1_to_I1"] = pc["O1_to_J3"] = pc["J1_to_K3"] = pc["I3_to_O3"]
###########################################################
pc["K3_to_O1"] = pc["O3_to_K1"] = pc["J3_to_I3"] = pc["I1_to_J1"]
pc["K3_to_O2"] = pc["O3_to_K2"] = pc["J3_to_I2"] = pc["I1_to_J2"]
pc["K3_to_O3"] = pc["O3_to_K3"] = pc["J3_to_I1"] = pc["I1_to_J3"]
pc["K3_to_J1"] = pc["O3_to_I3"] = pc["J3_to_O1"] = pc["I1_to_K1"]
pc["K3_to_J2"] = pc["O3_to_I2"] = pc["J3_to_O2"] = pc["I1_to_K2"]
pc["K3_to_J3"] = pc["O3_to_I1"] = pc["J3_to_O3"] = pc["I1_to_K3"]
pc["K3_to_I3"] = pc["O3_to_J1"] = pc["J3_to_K1"] = pc["I1_to_O1"]
pc["K3_to_I2"] = pc["O3_to_J2"] = pc["J3_to_K2"] = pc["I1_to_O2"]
pc["K3_to_I1"] = pc["O3_to_J3"] = pc["J3_to_K3"] = pc["I1_to_O3"]
###########################################################
pc["K2_to_O1"] = pc["O2_to_K1"] = pc["J2_to_I3"] = pc["I2_to_J1"]
pc["K2_to_O2"] = pc["O2_to_K2"] = pc["J2_to_I2"] = pc["I2_to_J2"]
pc["K2_to_O3"] = pc["O2_to_K3"] = pc["J2_to_I1"] = pc["I2_to_J3"]
pc["K2_to_J1"] = pc["O2_to_I3"] = pc["J2_to_O1"] = pc["I2_to_K1"]
pc["K2_to_J2"] = pc["O2_to_I2"] = pc["J2_to_O2"] = pc["I2_to_K2"]
pc["K2_to_J3"] = pc["O2_to_I1"] = pc["J2_to_O3"] = pc["I2_to_K3"]
pc["K2_to_I3"] = pc["O2_to_J1"] = pc["J2_to_K1"] = pc["I2_to_O1"]
pc["K2_to_I2"] = pc["O2_to_J2"] = pc["J2_to_K2"] = pc["I2_to_O2"]
pc["K2_to_I1"] = pc["O2_to_J3"] = pc["J2_to_K3"] = pc["I2_to_O3"]
###########################################################
action_node_dict = {}
action_node_dict[LEFT1]='K1'
action_node_dict[LEFT2]='K2'
action_node_dict[LEFT3]='K3'
action_node_dict[UP1]='J1'
action_node_dict[UP2]='J2'
action_node_dict[UP3]='J3'
action_node_dict[RIGHT1]='O1'
action_node_dict[RIGHT2]='O2'
action_node_dict[RIGHT3]='O3'
action_node_dict[DOWN1]='I1'
action_node_dict[DOWN2]='I2'
action_node_dict[DOWN3]='I3'
def noisy(noise_typ,image):
    if noise_typ == "gauss":
      row,col= image.shape
      mean = 0
      var = 0.01
      sigma = var**0.6
      gauss = np.random.uniform(low = mean,high = sigma,size = row*col)
      gauss = gauss.reshape(row,col)
      noisy = image + gauss
      return noisy
    elif noise_typ == "s&p":
      row,col = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
    elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    elif noise_typ =="speckle":
      row,col = image.shape
      gauss = np.random.randn(row,col)
      gauss = gauss.reshape(row,col)
      noisy = image + image * gauss
      return noisy


# maze is a 2d Numpy array of floats between 0.0 to 1.0
# 1.0 corresponds to a free cell, and 0.0 an occupied cell
# rat = (row, col) initial rat position (defaults to (0,0))

class Qmaze(object):
    def __init__(self, maze, rat=(0, 1)):
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        self.target = (nrows - 1, ncols - 2)  # target cell where the "cheese" is
        self.free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == 1.0]
        self.free_target_cells = []
        for r in range(nrows):
            for c in range(ncols):
                if self._maze[r, c] == 1.0:
                    if r == 0 or r == nrows-1 or c == 0 or c == ncols-1:
                        self.free_target_cells.append((r,c))
                    # if r == nrows - 1:
                    #     self.free_target_cells.append((r, c))
        self.free_cells.remove(self.target)
        self.current_distance = None
        self.available_cell_target = []
        self.centroid = []
        self.supervised = 0
        self.old_action = 1
        self.current_action = None
        self.num_non_available_cell = 32
        self.state_dim = self.num_non_available_cell * 2 + 4
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if not rat in self.free_cells:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        self.non_available_cell = random.choices(self.free_cells, k=self.num_non_available_cell)
        self.reset(rat, self.target, self.non_available_cell)

    def get_list_centroid(self):
        dx = [0, 2, 0, -2, -2, -1, 2, 1, -2, 1, 2, -1]
        dy = [2, 0, -2, 0, -1, 2, 1, -2, 1, 2, -1, -2]
        for i in range(self._maze.shape[0]):
            for j in range(self._maze.shape[1]):
                if (self._maze[i][j] == 0.95):
                    self.centroid.append((i, j))
                    for temp in range(12):
                        tx = i + dx[temp]
                        ty = j + dy[temp]
                        state_centroid_dict[(tx, ty)] = (i, j)

    def available_cell_for_agent(self):
        list_available_cell_agent = []
        nrows, ncols = self._maze.shape
        for row in range(nrows):
            for col in range(ncols):
                if row == 1 or row == 2 or row == 3 or row == 6 or row == 7 or row == 8 or row == 11 or row == 12 or row == 13:
                    if col == 0 or col == 14:
                        list_available_cell_agent.append((row, col))
                if col == 1 or col == 2 or col == 3 or col == 6 or col == 7 or col == 8 or col == 11 or col == 12 or col == 13:
                    if row == 0 or row == 14:
                        list_available_cell_agent.append((row, col))
                    if row == nrows - 1:
                        self.available_cell_target.append((row, col))
        return list_available_cell_agent

    def reset(self, rat, target, non_available_cell):
        self.rat = rat
        self._maze = get_map()
        nrows, ncols = self._maze.shape
        row, col = rat
        self._maze[row, col] = rat_mark
        self.state = (row, col, 'valid')
        self.min_reward = -256
        self.total_reward = 0
        self.visited = set()
        self.target = target
        target_row, target_col = self.target
        self.old_distance = -999
        self.list_previous_state = []
        self.non_available_cell = non_available_cell
        return self.observe().reshape(1, self.state_dim)

    def check_state_centroid(self):
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state

    def update_state(self, action):
        nrows, ncols = self._maze.shape
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state
        if self._maze[rat_row, rat_col] > 0:
            self.visited.add((rat_row, rat_col))  # mark visited cell

        valid_actions = self.valid_actions()
        x_centroid, y_centroid = state_centroid_dict[(nrow, ncol)]
        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            dx = [0, 2, 0, -2, -2, -1, 2, 1, -2, 1, 2, -1]
            dy = [2, 0, -2, 0, -1, 2, 1, -2, 1, 2, -1, -2]
            # r2,d2,l2,u2,u1,r1,d3,l3,u3,r3,d1,l1
            # 0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10 ,11
            if action == LEFT1:
                if self._maze[nrow][ncol - 1] == 1 and self._maze[nrow][ncol + 1] == 0.9:
                    x_centroid += dx[11]
                    y_centroid += (dy[11] - 1)
                    nmode = 'reward_unchanged'
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
                else:
                    x_centroid += dx[11]
                    y_centroid += dy[11]
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
            elif action == LEFT2:
                if self._maze[nrow][ncol - 1] == 1 and self._maze[nrow][ncol + 1] == 0.9:
                    x_centroid += dx[2]
                    y_centroid += (dy[2] - 1)
                    nmode = 'reward_unchanged'
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
                else:
                    x_centroid += dx[2]
                    y_centroid += dy[2]
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
            elif action == LEFT3:
                if self._maze[nrow][ncol - 1] == 1 and self._maze[nrow][ncol + 1] == 0.9:
                    x_centroid += dx[7]
                    y_centroid += (dy[7] - 1)
                    nmode = 'reward_unchanged'
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
                else:
                    x_centroid += dx[7]
                    y_centroid += dy[7]
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
            elif action == RIGHT1:
                if self._maze[nrow][ncol + 1] == 1 and self._maze[nrow][ncol - 1] == 0.9:
                    x_centroid += dx[5]
                    y_centroid += (dy[5] + 1)
                    nmode = 'reward_unchanged'
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
                else:
                    x_centroid += dx[5]
                    y_centroid += dy[5]
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
            elif action == RIGHT2:
                if self._maze[nrow][ncol + 1] == 1 and self._maze[nrow][ncol - 1] == 0.9:
                    x_centroid += dx[0]
                    y_centroid += (dy[0] + 1)
                    nmode = 'reward_unchanged'
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
                else:
                    x_centroid += dx[0]
                    y_centroid += dy[0]
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
            elif action == RIGHT3:
                if self._maze[nrow][ncol + 1] == 1 and self._maze[nrow][ncol - 1] == 0.9:
                    x_centroid += dx[9]
                    y_centroid += (dy[9] + 1)
                    nmode = 'reward_unchanged'
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
                else:
                    x_centroid += dx[9]
                    y_centroid += dy[9]
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
            elif action == DOWN1:
                if self._maze[nrow + 1][ncol] == 1 and self._maze[nrow - 1][ncol] == 0.9:
                    x_centroid += (dx[10] + 1)
                    y_centroid += dy[10]
                    nmode = 'reward_unchanged'
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
                else:
                    x_centroid += dx[10]
                    y_centroid += dy[10]
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
            elif action == DOWN2:
                if self._maze[nrow + 1][ncol] == 1 and self._maze[nrow - 1][ncol] == 0.9:
                    x_centroid += (dx[1] + 1)
                    y_centroid += dy[1]
                    nmode = 'reward_unchanged'
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
                else:
                    x_centroid += dx[1]
                    y_centroid += dy[1]
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
            elif action == DOWN3:
                if self._maze[nrow + 1][ncol] == 1 and self._maze[nrow - 1][ncol] == 0.9:
                    x_centroid += (dx[6] + 1)
                    y_centroid += dy[6]
                    nmode = 'reward_unchanged'
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
                else:
                    x_centroid += dx[6]
                    y_centroid += dy[6]
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
            elif action == UP1:
                if self._maze[nrow - 1][ncol] == 1 and self._maze[nrow + 1][ncol] == 0.9:
                    x_centroid += (dx[4] - 1)
                    y_centroid += dy[4]
                    nmode = 'reward_unchanged'
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
                else:
                    x_centroid += dx[4]
                    y_centroid += dy[4]
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
            elif action == UP2:
                if self._maze[nrow - 1][ncol] == 1 and self._maze[nrow + 1][ncol] == 0.9:
                    x_centroid += (dx[3] - 1)
                    y_centroid += dy[3]
                    nmode = 'reward_unchanged'
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
                else:
                    x_centroid += dx[3]
                    y_centroid += dy[3]
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
            elif action == UP3:
                if self._maze[nrow - 1][ncol] == 1 and self._maze[nrow + 1][ncol] == 0.9:
                    x_centroid += (dx[8] - 1)
                    y_centroid += dy[8]
                    nmode = 'reward_unchanged'
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
                else:
                    x_centroid += dx[8]
                    y_centroid += dy[8]
                    if self._maze[x_centroid, y_centroid] == 0.0:
                        nmode = 'blocked'
                    else:
                        self.state = (x_centroid, y_centroid, nmode)
        else:  # invalid action, no change in rat position
            # nmode = 'invalid'
            pass

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        target_row, target_col = self.target
        nrows, ncols = self._maze.shape
        reward = None
        status = None
        if mode == 'reward_unchanged' and not (rat_row, rat_col) in self.visited:
            # mode = "valid"
            return 0
        #         if self.current_action == self.old_action :
        #           return -1.2
        if mode == 'blocked':
            reward = self.min_reward - 0.5
        if (rat_row, rat_col) in self.visited:
            return -1.5
        if mode == 'invalid':
            reward = -2
        # if mode == 'valid':
        #     reward  = -0.04
        if mode == 'valid':
            # self.current_distance = self.dist(rat_row,rat_col,target_row,target_col)
            # if self.current_distance <= self.old_distance :
            #       reward = -self.current_distance*0.004

            # else : reward = -self.current_distance*0.005
            # self.old_distance = self.current_distance
            try:
                reward = df["{}_to_{}".format(action_node_dict[self.old_action], action_node_dict[self.current_action])]
            except:
                reward = -1
        if rat_row == target_row and rat_col == target_col:
            status = 'win'
            return 2
        # if mode == 'valid' and status !='win':
        #       self.current_distance = self.dist(rat_row,rat_col,target_row,target_col)
        #       if (rat_row, rat_col) in self.visited :
        #           reward = -0.25
        #       elif self.current_distance >= self.old_distance :
        #       #reward = -current_distance*0.001
        #           reward = -0.1
        #       else : reward = -0.04
        #       self.old_distance = self.current_distance
        return reward

    def dist(self, x1, y1, x2, y2):
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def act_to_predict(self, action):
        self.update_state_to_predict(action)
        status = self.game_status_to_predict()
        envstate = self.observe()
        return envstate, status

    def observe(self):
        canvas = self.draw_env(self.non_available_cell)
        envstate = np.reshape(canvas, newshape=(1, self.state_dim))
        return envstate

    def draw_env(self, non_available_cell):
        self._maze = get_map()
        row, col = self.rat
        self._maze[row, col] = rat_mark
        canvas = np.copy(self._maze)
        nrows, ncols = self._maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0
                if (r, c) in non_available_cell:
                    canvas[r, c] = 0.0
                    self._maze[r, c] = 0.0
        # draw the rat
        for row, col in self.visited:
            canvas[row, col] = 0.6
        row, col, valid = self.state
        row_target, col_target = self.target
        canvas[row, col] = rat_mark
        canvas[row_target, col_target] = target_mark
        # canvas = noisy("gauss",canvas)
        input_state = [row, col, row_target, col_target]
        for item in non_available_cell:
            input_state.append(item[0])
            input_state.append(item[1])
        input_state = np.asarray(input_state).reshape(-1, 1)
        std_scale = preprocessing.StandardScaler().fit(input_state)
        input_state = std_scale.transform(input_state)
        return input_state

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        rat_row, rat_col, mode = self.state
        nrows, ncols = self._maze.shape
        target_row, target_col = self.target
        if rat_row == target_row and rat_col == target_col:
            return 'win'

        return 'not_over'

    def set_supervised(self, supervised):
        self.supervised = supervised

    def game_status_to_predict(self):
        rat_row, rat_col, mode = self.state
        target_row, target_col = self.target
        if rat_row == target_row and rat_col == target_col:
            return 'win'

        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [LEFT1, LEFT2, LEFT3, RIGHT1, RIGHT2, RIGHT3, UP1, UP2, UP3, DOWN1, DOWN2, DOWN3]
        nrows, ncols = self._maze.shape
        if row == 0:
            actions.remove(UP1)
            actions.remove(UP2)
            actions.remove(UP3)
        elif row == nrows - 1:
            actions.remove(DOWN1)
            actions.remove(DOWN2)
            actions.remove(DOWN3)
        if col == 0:
            actions.remove(LEFT1)
            actions.remove(LEFT2)
            actions.remove(LEFT3)
        elif col == ncols - 1:
            actions.remove(RIGHT1)
            actions.remove(RIGHT2)
            actions.remove(RIGHT3)
        # if row>0 and self.maze[row-1,col] == 0.0:
        #     actions.remove(UP1)
        #     actions.remove(UP2)
        #     actions.remove(UP3)
        #     actions.remove(UP4)
        # if row<nrows-1 and self.maze[row+1,col] == 0.0:
        #     actions.remove(DOWN1)
        #     actions.remove(DOWN2)
        #     actions.remove(DOWN3)
        #     actions.remove(DOWN4)
        # if col>0 and self.maze[row,col-1] == 0.0:
        #     actions.remove(LEFT1)
        #     actions.remove(LEFT2)
        #     actions.remove(LEFT3)
        #     actions.remove(LEFT4)
        # if col<ncols-1 and self.maze[row,col+1] == 0.0:
        #     actions.remove(RIGHT1)
        #     actions.remove(RIGHT2)
        #     actions.remove(RIGHT3)
        #     actions.remove(RIGHT4)
        dx = [0, 2, 0, -2, -2, -1, 2, 1, -2, 1, 2, -1]
        dy = [2, 0, -2, 0, -1, 2, 1, -2, 1, 2, -1, -2]
        # r2,d2,l2,u2,u1,r1,d3,l3,u3,r3,d1,l1
        # 0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10 ,11
        tx_target, ty_target = self.target
        if self.supervised == 1:
            best_choice = None
            best_distance = 999999
            for action in actions:
                x_centroid, y_centroid = state_centroid_dict[(row, col)]
                if action == LEFT1:
                    if self._maze[row][col - 1] == 1 and self._maze[row][col + 1] == 0.9:
                        x_centroid += dx[11]
                        y_centroid += (dy[11] - 1)
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    else:
                        x_centroid += dx[11]
                        y_centroid += dy[11]
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    if self._maze[x_centroid, y_centroid] == 0:
                        cur_distance = 9999
                elif action == LEFT2:
                    if self._maze[row][col - 1] == 1 and self._maze[row][col + 1] == 0.9:
                        x_centroid += dx[2]
                        y_centroid += (dy[2] - 1)
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    else:
                        x_centroid += dx[2]
                        y_centroid += dy[2]
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    if self._maze[x_centroid, y_centroid] == 0:
                        cur_distance = 9999
                elif action == LEFT3:
                    if self._maze[row][col - 1] == 1 and self._maze[row][col + 1] == 0.9:
                        x_centroid += dx[7]
                        y_centroid += (dy[7] - 1)
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    else:
                        x_centroid += dx[7]
                        y_centroid += dy[7]
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    if self._maze[x_centroid, y_centroid] == 0:
                        cur_distance = 9999
                elif action == RIGHT1:
                    if self._maze[row][col + 1] == 1 and self._maze[row][col - 1] == 0.9:
                        x_centroid += dx[5]
                        y_centroid += (dy[5] + 1)
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    else:
                        x_centroid += dx[5]
                        y_centroid += dy[5]
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    if self._maze[x_centroid, y_centroid] == 0:
                        cur_distance = 9999
                elif action == RIGHT2:
                    if self._maze[row][col + 1] == 1 and self._maze[row][col - 1] == 0.9:
                        x_centroid += dx[0]
                        y_centroid += (dy[0] + 1)
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    else:
                        x_centroid += dx[0]
                        y_centroid += dy[0]
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    if self._maze[x_centroid, y_centroid] == 0:
                        cur_distance = 9999
                elif action == RIGHT3:
                    if self._maze[row][col + 1] == 1 and self._maze[row][col - 1] == 0.9:
                        x_centroid += dx[9]
                        y_centroid += (dy[9] + 1)
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    else:
                        x_centroid += dx[9]
                        y_centroid += dy[9]
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    if self._maze[x_centroid, y_centroid] == 0:
                        cur_distance = 9999
                elif action == DOWN1:
                    if self._maze[row + 1][col] == 1 and self._maze[row - 1][col] == 0.9:
                        x_centroid += (dx[10] + 1)
                        y_centroid += dy[10]
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    else:
                        x_centroid += dx[10]
                        y_centroid += dy[10]
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    if self._maze[x_centroid, y_centroid] == 0:
                        cur_distance = 9999
                elif action == DOWN2:
                    if self._maze[row + 1][col] == 1 and self._maze[row - 1][col] == 0.9:
                        x_centroid += (dx[1] + 1)
                        y_centroid += dy[1]
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    else:
                        x_centroid += dx[1]
                        y_centroid += dy[1]
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    if self._maze[x_centroid, y_centroid] == 0:
                        cur_distance = 9999
                elif action == DOWN3:
                    if self._maze[row + 1][col] == 1 and self._maze[row - 1][col] == 0.9:
                        x_centroid += (dx[6] + 1)
                        y_centroid += dy[6]
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    else:
                        x_centroid += dx[6]
                        y_centroid += dy[6]
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    if self._maze[x_centroid, y_centroid] == 0:
                        cur_distance = 9999
                elif action == UP1:
                    if self._maze[row - 1][col] == 1 and self._maze[row + 1][col] == 0.9:
                        x_centroid += (dx[4] - 1)
                        y_centroid += dy[4]
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    else:
                        x_centroid += dx[4]
                        y_centroid += dy[4]
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    if self._maze[x_centroid, y_centroid] == 0:
                        cur_distance = 9999
                elif action == UP2:
                    if self._maze[row - 1][col] == 1 and self._maze[row + 1][col] == 0.9:
                        x_centroid += (dx[3] - 1)
                        y_centroid += dy[3]
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    else:
                        x_centroid += dx[3]
                        y_centroid += dy[3]
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    if self._maze[x_centroid, y_centroid] == 0:
                        cur_distance = 9999
                elif action == UP3:
                    if self._maze[row - 1][col] == 1 and self._maze[row + 1][col] == 0.9:
                        x_centroid += (dx[8] - 1)
                        y_centroid += dy[8]
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    else:
                        x_centroid += dx[8]
                        y_centroid += dy[8]
                        cur_distance = self.dist(x_centroid, y_centroid, tx_target, ty_target)
                    if self._maze[x_centroid, y_centroid] == 0:
                        cur_distance = 9999
                if (cur_distance < best_distance):
                    best_distance = cur_distance
                    best_choice = action
                # print("Best distance : ",best_distance, " Best choice : ",best_choice)
            actions.clear()
            actions.append(best_choice)
        return actions

    def check_valid_with_previous_state(self, cell):
        row_cell, col_cell = cell
        for temp in self.visited:
            row, col = temp
            if row_cell == row and col_cell == col:
                return False
        return True
def show(qmaze):
    plt.grid('on')
    nrows, ncols = qmaze._maze.shape
    row_target,col_target = qmaze.target
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows+20, 1))
    ax.set_yticks(np.arange(0.5, ncols+20, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze._maze)
    for row,col in qmaze.visited:
        canvas[row,col] = 0.7
    rat_row, rat_col, _ = qmaze.state
    canvas[rat_row, rat_col] = 0.3   # rat cell
    canvas[row_target,col_target] = 0.2 # cheese cell
    img = plt.imshow(canvas, interpolation='none', cmap='Reds_r')
    plt.show()
    return img
img_width, img_height = maze.shape
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_height, img_width)
else:
    input_shape = (img_height,img_width, 1)

def dist(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        self.writer.reopen()
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)
        self.writer.flush()

