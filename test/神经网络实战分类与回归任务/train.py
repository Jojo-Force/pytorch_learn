import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import warnings

from pyexpat import features

warnings.filterwarnings("ignore")
#%matplotlib inline
features = pd.read_csv('temps.csv')

a=features
print(a)