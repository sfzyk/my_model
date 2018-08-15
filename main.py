from torchsummary import summary
from Model import MyModel

__author__ = "Li Xi"

n_head = 8
d_model = 512
d_k = 64
d_v = 64
n_block = 8
dropout = 0.1
num_class = 3

model = MyModel(n_head, d_model, d_k, d_v, n_block, num_class, dropout=0.1)

summary(model, (100, 512))
