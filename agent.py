import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam 
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical
import os
#Sub classes
from actor import Actor
from critic import Critic
from keras.models import load_model
from advisor import Advisor
from seb import *
class Agent:

       
    def __init__(self,input_dim, output_dim, lr, gamma, loss_clipping, c1, lamda, k_permutation):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actions = range(output_dim)  
        self.lr = lr
        self.gamma = gamma
        self.lamda = lamda
        self.k_permutation = k_permutation
        self.loss_clipping = loss_clipping  #for actor loss function
        self.c1 = c1   #weight for entropy term in actor loss function
        self.num_epochs = 10
        self.batchsize = 16
        #These will store the samples from which the agent will learn
        self.states = []
        self.actions = []
        self.pi_vecs = []
        self.rewards = []
        self.masks = []
        #Make actor and critic
        self.actor = Actor(input_dim,output_dim,lr,gamma,loss_clipping,c1)
        self.advisor = Advisor(input_dim,output_dim,lr,gamma,loss_clipping,c1)
        self.critic = Critic(input_dim, output_dim, self.lr)
        self.seb = SEB(k_permutation)
        try:
            print("Loading model...")
            self.load_model(True)
        except:
            print("Load model failed")
        
    def check_successful_sample(self,S,A,masks,R):
        state_sample = []
        action_sample = []
        reward_sample = []
        reach_target_value = False
        reach_goal_in_step = np.where(masks == reach_target_value)
        init_term = 0
        for i in range (reach_goal_in_step.size()):
            for j in range (init_term, reach_goal_in_step[i]+1):
                state_sample.append(S[j])
                action_sample.append(A[j])
                reward_sample.append(R[j])
            self.seb.get_sample(state_sample, action_sample, reward_sample)
            init_term = reach_goal_in_step[i]+2




    def get_batch(self):
        """ For now, just take all the thing in memory """
        
        #Turn lists into arrays
        S = np.array(self.states)             #stack of state vectors
        A = np.array(self.actions)     # stack of one-hot action vectors
        masks = np.array(self.masks)
        Pi = np.array(self.pi_vecs)           # stack of pi_vec, where pi_vec_i = \pi(s_i, a_i)
        R = np.array(self.rewards)            # stack of rewards vecs (a vector with 1 element = scalar)
        self.check_successful_sample(S,A,masks,R)
        return S,A,masks,Pi,R
    
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.masks = []
        self.pi_vecs = []
        self.rewards = []
    
    
    def train_models(self):
    
        #Easist implementation of batching, will make more sophisticated later
        states, actions,masks, pi_vecs, rewards = self.get_batch()
        self.num_epochs = len(states)*2//self.batchsize
        self.clear_memory()
        states = np.reshape(states, (-1, self.input_dim))
        values = self.critic.model.predict(states)
        values = np.asarray(values)
        values.resize(len(values))
        #Compute inputs for the optimizers
        discounted_return, advantages = self.get_advantages(values,masks,rewards)
        # print(discounted_return)
        # print(values)
        #Do the training
        old_predictions = pi_vecs
        actor_loss = self.actor.model.fit([states, advantages, old_predictions,np.reshape(discounted_return, newshape=(-1, 1, 1)),np.reshape(values, newshape=(-1, 1, 1))], [actions], batch_size=self.batchsize, shuffle=True, epochs=self.num_epochs, verbose=1)
        critic_loss = self.critic.model.fit([states], [discounted_return], \
                      batch_size=self.batchsize, shuffle=True, epochs=self.num_epochs, verbose=1)

        


    def get_advantages(self,values, masks, rewards):
        discounted_returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            if i==len(rewards)-1 :
                delta = rewards[i] - values[i]
            else :
                delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lamda * masks[i] * gae
            discounted_returns.insert(0, gae + values[i])
        discounted_returns = np.asarray(discounted_returns)
        discounted_returns = discounted_returns.astype('float32')
        discounted_returns -= np.mean(discounted_returns)
        discounted_returns /= (np.std(discounted_returns) + 1e-10)
        advantages = discounted_returns - values
        return discounted_returns, advantages
    
        
    
    def remember(self, state, action,mask, pi_vec, reward):
        self.states.append(state)
        
        action_onehot = to_categorical(action,self.output_dim) #optimizers use one-hot
        self.actions.append(action_onehot)
        self.masks.append(mask)
        
        self.pi_vecs.append(pi_vec)
        self.rewards.append(reward)
        

    def act(self, state):
        """ Choose action according to softmax """

        num_samples = state.shape[0]
        adv_dummy = np.zeros((num_samples, 1))
        pi_old_dummy = np.zeros((num_samples, self.output_dim))
        dummy_reward = np.zeros((1, 1, 1))
        dummy_value = np.zeros((1, 1, 1))
        inp = [state, adv_dummy, pi_old_dummy,dummy_reward,dummy_value]        
        #Find action
        pi_vec = self.actor.model.predict(inp)[0]      # prob of actions
        action = np.random.choice(range(self.output_dim), p=pi_vec)
        return action, pi_vec
    
    
    def soft_update_target_network(self,net):
        
        pars_behavior = net.model.get_weights()       # these have form [W1, b1, W2, b2, ..], Wi = 
        pars_target = net.target_model.get_weights()  # bi = biases in layer i
        
        ctr = 0
        for par_behavior,par_target in zip(pars_behavior,pars_target):
            par_target = par_target*(1-self.tau) + par_behavior*self.tau
            pars_target[ctr] = par_target
            ctr += 1

        net.target_model.set_weights(pars_target)
        
        
    def save_target_weights(self):

        """ Saves the weights of the target 
            network (only use the target during
            testing, so don't need to save tje
            behavior)
        """

        #Create directory if it doesn't exist
        dir_name = 'network_weights/'
        if not os.path.exists(os.path.dirname(dir_name)):
            os.makedirs(os.path.dirname(dir_name))

        #Now save the weights. I'm choosing ID by gamma, lr, tau
        if self.seed_num == False:
            pars_tag = '_gamma_' + str(self.gamma) + '_lr_' + str(self.lr) + '_tau_' + str(self.tau)
        else:
            pars_tag = '_gamma_' + str(self.gamma) + '_lr_' + str(self.lr) + '_tau_' + str(self.tau) \
            +'_seed_' + str(self.seed_num)
            

        #Actor target network
        filename = 'network_weights/actor_target'
        actor_pars = self.actor.target_model.get_weights()
        np.save(filename + pars_tag, actor_pars)

        #Critic target network
        filename = 'network_weights/critic_target'
        critic_pars = self.critic.target_model.get_weights()
        np.save(filename + pars_tag, critic_pars)




    def load_target_weights(self,gamma,lr,tau):

        """ Loads the weights of the target 
            network, previously created using
            the save_target_wieghts() function
        """


        #Now save the weights. I'm choosing ID by gamma, lr, tau
        if self.seed_num == False:
            pars_tag = '_gamma_' + str(self.gamma)+'_lr_'+str(self.lr)+'_tau_' + str(self.tau) + '.npy'
        else:
            pars_tag = '_gamma_' + str(self.gamma)+'_lr_'+str(self.lr)+'_tau_'+str(self.tau)+'_seed_' \
            +str(self.seed_num)+ '.npy'

        #Actor target network
        filename = 'network_weights/actor_target'
        actor_pars = np.load(filename + pars_tag)
        self.actor.target_model.set_weights(actor_pars)

        #Critic target network
        filename = 'network_weights/critic_target'
        critic_pars = np.load(filename + pars_tag)
        self.critic.target_model.set_weights(critic_pars)

    def save_model(self):
        self.actor.model.save_weights("actor.h5")
        self.critic.model.save_weights("critic.h5")

    def load_model(self, mode):
        if mode == True:
            self.actor.model.load_weights("actor.h5")
            self.critic.model.load_weights("critic.h5")