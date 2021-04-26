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


class Actor:

    
    def __init__(self,input_dim, output_dim,lr,gamma,loss_clipping,c1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr  #learning rate for optimizer
        self.gamma = gamma
        self.loss_clipping = loss_clipping
        self.c1 = c1  #constant for entropy loss
        self.num_layers = 3
        self.hidden_dim = 512
        self.model = self._make_network()
        
        #No target models for actor
        #self.target_model = self._make_network()                # target networks to stabilize learning.
        #self.target_model.set_weights(self.model.get_weights()) # clone the networks
        
        
    def proximal_policy_optimization_loss(self,advantage,old_prediction,rewards,values):
        def loss(y_true, y_pred):
            
            #L_{CPI}
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob/(old_prob + 1e-10)
            term1 = r * advantage
            term2 = K.clip(r, min_value=1 - self.loss_clipping, max_value=1 + self.loss_clipping)*advantage
            loss_CPI  = K.minimum(term1, term2)
            critic_loss = 0.5*K.mean(K.square(rewards - values))
            #L_entropy
            loss_entropy = 0.01*self.c1*(prob * K.log(prob + 1e-10))
    
            return -K.mean(loss_CPI + loss_entropy) + critic_loss
        
        return loss
        
        
    def _make_network(self):
        
        #Inputs
        state_input = Input(shape=(self.input_dim,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(self.output_dim,))
        rewards = Input(shape=(1, 1,))
        values = Input(shape=(1, 1,))
        #NN
        x = Dense(self.hidden_dim, activation='elu')(state_input)
        for _ in range(self.num_layers - 1):
            x = Dense(self.hidden_dim, activation='elu')(x)

        out_actions = Dense(self.output_dim, activation='softmax', name='output')(x)
        
        #I will make the model the take three inputs, state, adv, old_prediction, as
        #as opposed to just a state. This is to make it compatible with the loss fucntion
        model = Model(inputs=[state_input, advantage, old_prediction,rewards,values], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=self.lr),
                      loss=[self.proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction=old_prediction,
                          rewards=rewards,values=values)])
        import time
        model.summary()
        time.sleep(5)
        return model