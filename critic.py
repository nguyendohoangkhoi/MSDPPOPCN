import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential, Input, Model
from keras.layers import Dense
from keras.optimizers import Adam 
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical


class Critic:

    
    def __init__(self,input_dim, output_dim,lr):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr  #learning rate for optimizer
        self.loss_clipping = 0.2
        self.c1 = 0.001  #constant for entropy loss
        self.num_layers = 3
        self.hidden_dim = 512
        self.model = self._make_network()
        
        #No target models in PPO
        #self.target_model = self._make_network()                # target networks to stabilize learning.
        #self.target_model.set_weights(self.model.get_weights()) # clone the networks
        
        
    def _make_network(self):
        state_input = Input(shape=(self.input_dim,))
        x = Dense(self.hidden_dim, activation='elu')(state_input)
        for _ in range(self.num_layers - 1):
            x = Dense(self.hidden_dim, activation='elu')(x)
        out_value = Dense(1)(x)
        
        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=self.lr), loss='mse')
        import time
        model.summary()
        time.sleep(5)
        return model