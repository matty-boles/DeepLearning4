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

random_tensor = torch.rand(size = (2,2,3,4))
print(torch.numel(random_tensor))

#Create a random tensor with similar shape to an image tensor
#Images have height, width, colour channels

random_image_size_tensor = torch.rand(size=(224, 224, 3))
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)




