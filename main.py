import numpy as np
import torch
from torchvision import models
from torchsummary import summary

from Model import MyModel

n_head = 8
d_model = 512
d_k = 64
d_v = 64

dropout=0.1


model = MyModel(n_head, d_model, d_k, d_v, dropout=0.1)

summary(model, (10,512))



