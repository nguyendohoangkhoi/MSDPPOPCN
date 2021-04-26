import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from keras.models import Sequential, Input, Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical
import collections

class SEB:

    def __init__(self, k_permutation, sizeMemory):
        self.k_permutation = k_permutation
        self.memory = []
        self.target_memory = collections.defaultdict(list)
        self.memorySize = sizeMemory

    def permutate_sample(self, samples):
        x_target, y_target = samples[0][2],samples[0][3]
        series_sample_from_memory = self.target_memory["{}_{}".format(x_target,y_target)]
        for i in range(len(series_sample_from_memory)):
            pass



    def get_sample(self, state_samples, action_samples, reward_samples ):
        x_target, y_target = state_samples[0][2], state_samples[0][3]
        self.target_memory["{}_{}".format(x_target,y_target)].append([state_samples, action_samples, reward_samples])
        self.memory.append([state_samples, action_samples, reward_samples])


