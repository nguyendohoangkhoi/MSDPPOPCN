# Multi-Sample Discovery Proximal Policy Optimization
Our data is included in "data" folder. In a switch, it has four directions corresponding to the I,J,O,K. Each direction has three switches, for example I1,I2,I3. Therefore, the format name of those data are "NamePort_to_NamePort".
# Data Format
![image](https://user-images.githubusercontent.com/37859108/117179029-e200d000-adfc-11eb-843a-e56c46e580a8.png)

Here, we surveyed the PCN with the wavelength in range from 1.525 to 1.565 (first collum). The output (tranmission), loss_i (crosstalk) collum is to compute the reward which is tranmission loss for the agent. The pc collum is to computer power consumption for each switching. The equation (17) to compute tranmission loss is clearly defined in the paper.
# Code Usage
For more detail about our implementation, please go to notebook file (RLA-Standard.ipynb) to follow step by step. We describe how to implement the map of PCN including change map size, define reward, etc. 
