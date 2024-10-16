#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from termcolor import colored

#%%
# Check if Pytorch is using the GPU:
print(colored("should be true: ", "yellow"), torch.cuda.is_available()) #should be true
print(colored("should be 1: ","yellow"),torch.cuda.device_count()) #should be 1
print(colored("should be 0: ","yellow"),torch.cuda.current_device()) #should be 0
print(colored("GPU name: ","yellow"),torch.cuda.get_device_name(0)) #should return the GPU name

#%%
