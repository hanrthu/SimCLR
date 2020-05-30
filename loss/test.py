import torch 
import numpy as np

batch_size = 200
A = torch.rand(batch_size,batch_size)
i = np.arange(batch_size)
x = np.random.choice(batch_size,batch_size)
print(x.shape)
A = A[i, x]
print(A.shape[0])