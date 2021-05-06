# Multi-Sample Discovery Proximal Policy Optimization
Our data is included in "data" folder. In a switch, it has four directions corresponding to the I,J,O,K. Each direction has three switches, for example I1,I2,I3. Therefore, the format name of those data are "NamePort_to_NamePort".
# Data Format
![image](https://user-images.githubusercontent.com/37859108/117179029-e200d000-adfc-11eb-843a-e56c46e580a8.png)

Here, we surveyed the PCN with the wavelength in range from 1.525 to 1.565 (first collum). The output (tranmission), loss_i (crosstalk) collum is to compute the reward which is tranmission loss for the agent. The pc collum is to computer power consumption for each switching. The equation (17) to compute tranmission loss is clearly defined in the paper.
# Code Usage
The model of actor, advisor, critic and seb are implemented in actor.py, advisor.py, critic.py and seb.py
For more detail about our implementation, please go to notebook file (RLA-Standard.ipynb) to follow step by step. We describe how to implement the map of PCN including changing map size, define reward, how the map work and the msd-ppo algorithm in PCN etc. 
If you want to check the training process. You can use tensorboard by this command "tensorboard --logdir=logs/"
