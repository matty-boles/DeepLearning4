import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Introduction to Tensors

### Creating Tensors

#Scalar
# scalar = torch.tensor(7)
# scalar = scalar.item()
# print (type(scalar))

#Vector
# vector = torch.tensor([7,7])
# print(vector)
# print(vector.ndim)
# print(vector.shape)

#MATRIX
# MATRIX = torch.tensor([[7,8],
#                        [9,10]])
# print(MATRIX)
# print(MATRIX.ndim)
# print(MATRIX[0])
# print(MATRIX.shape)

#Tensor

# TENSOR = torch.tensor([[[[1,2,3],
#             [5,6,7],
#             [8,9,10]],
#             [[1,2,3],
#             [5,6,7],
#             [8,9,10]]],
#             [[[1,2,3],
#             [5,6,7],
#             [8,9,10]],
#             [[1,2,3],
#             [5,6,7],
#             [8,9,10]]]])
# print(TENSOR.ndim)
# print(TENSOR.shape)
# print(TENSOR[0])

### Random Tensors

# random_tensor = torch.rand(size = (2,2,3,4))
# print(torch.numel(random_tensor))

# #Create a random tensor with similar shape to an image tensor
# #Images have height, width, colour channels

# random_image_size_tensor = torch.rand(size=(224, 224, 3))
# print(random_image_size_tensor.shape, random_image_size_tensor.ndim )

### Zeros and Ones

# zeros = torch.zeros(size = (3,4))
# print(zeros)

# ones = torch.ones(size=(3,4))
# print(ones.dtype)

### Range of tensors

#print(torch.arange(start=0,end=10.1,step=2))

### Tensors like

# MATRIX = torch.tensor([[[2,3],
#            [3,4],
#            [2,3],
#            [3,4]],
#            [[2,3],
#            [3,4],
#            [2,3],
#            [3,4]]])
# zeros = torch.zeros_like(input=MATRIX)
# print(zeros)

###Tensor datatypes

# float_32_tensor = torch.tensor([3,6,9],
#                                dtype=torch.float16,
#                                device = None,
#                                requires_grad=False)
# print(float_32_tensor)

###Getting info from Tensors

#some_tensor = torch.rand(size=(3,4))

# print(some_tensor)
# print(f"Datatype of data: {some_tensor.dtype}")
# some_tensor = some_tensor.type(torch.float16)
# print(f"Datatype of data: {some_tensor.dtype}")
# print(f"Shape of data: {some_tensor.shape}")
# print(f"Device of data: {some_tensor.device}")

### Manipulating Tensors

# tensor = torch.tensor([1,2,3])
# print(tensor + 10)
# print(tensor*10)
# print(tensor-10)
# print(tensor/3)

### Matrix Multiplication & Transpose
# tensor1 = torch.tensor([[1,2],
#                         [4,5]])
# tensor2 = torch.tensor([[1,2],
#                         [4,5],
#                         [2,3],
#                         [2,3]])
# tensor3 = torch.matmul(tensor1, tensor2.T)
# print(tensor3.shape[0])

### Tensor Aggregration

# x = torch.arange(start=0,end=101,step=10)
# print(torch.min(x))
# print(torch.max(x))
# print(torch.mean(x.type(torch.float32)))
# print(torch.sum(x))
# print(torch.argmin(x))
# print(torch.argmax(x))

### Reshaping, Stacking, Squeezing, Unsqueezing Tensors

# x = torch.arange(start=1, end=10)
# print(x, x.shape, x.ndim)

# #Add an extra dimension

# x_reshaped = x.reshape(3,3)
# print(x_reshaped, x_reshaped.shape, x_reshaped.ndim)

#Views
# z = x.view(3,3)
# print(z,z.shape)
# z[:,2] =11
# print(z)
# print(x)

# Stack Tensors
# x_stacked = torch.hstack([x,x,x,x])
# print(x_stacked)

# Squeeze - removes all dims of sizs 1 from a tensor

# x = x.reshape(1,9)
# print(x.shape)
# x_squeezed = x.squeeze()
# print(x_squeezed.shape)
# x_unsqueezed = x_squeezed.unsqueeze(dim = 1)
# print(x_unsqueezed)
# x_unsqueezed = x_unsqueezed.unsqueeze(dim = 2)
# print(x_unsqueezed, x_unsqueezed.shape)

# Permute - rearranges dims in a tensor to a specific order

# x = torch.randn(2,1,2)
# print(x)
# z = torch.permute(x, (2,1,0))
# print(z)
# z[0,0,0]= 0
# print(x)

### Numpy and Pytorch

# array = np.arange(1,8)
# tensor = torch.from_numpy(array)
# print(array, tensor)

# numpyArray = tensor.numpy()
# print(numpyArray)

### Reproducability

# random_tensor_A = torch.rand(size=(3,4))
# random_tensor_B = torch.rand(size=(3,4))
# print(random_tensor_A)
# print(random_tensor_B)
# print(random_tensor_A == random_tensor_B)

# RANDOM_SEED = 42
# torch.manual_seed(RANDOM_SEED)
# random_tensor_C = torch.rand(size=(3,4))
# torch.manual_seed(RANDOM_SEED)
# random_tensor_D = torch.rand(size=(3,4))
# print(random_tensor_C)
# print(random_tensor_D)
# print(random_tensor_C == random_tensor_D)







 










 


