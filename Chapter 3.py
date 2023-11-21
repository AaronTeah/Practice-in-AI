#!/usr/bin/env python
# coding: utf-8

# # Exponents

# In[1]:


m = 3
n = 4

p = m**n 
print("The answer is ", p)


# In[2]:


m = 3
n = 4

p = pow(m, n)
print("The answer is ", p)


# # Radical / Square Root

# In[3]:


import math

a = 49
b = 100

sqrta = math.sqrt(a)
sqrtb = math.sqrt(b)

print("The answer a is", sqrta)
print("The answer b is", sqrtb)


# # Factorials

# In[4]:


import math

a = 4
b = 7

print("Factorial of 4 is", math.factorial(a))
print("Factorial of 7 is", math.factorial(b))


# # Summation / Sigma Notation

# In[5]:


sum([1,2,3,4])


# In[6]:


def sum_numbers(numbers): 
    if len(numbers) == 0: 
        return 0
    return numbers[0] + sum_numbers(numbers[1:])

sum_numbers([1,2,3,4])


# # Vector and Vector Functions

# In[7]:


import numpy as np

lst = [10,20,30,40,50,60]

vctr = np.array(lst)

print("Vector created from a list:")
print(vctr)


# In[8]:


#Vertical Vector

lst = [[2], 
      [4], 
      [6], 
      [10]]

vctr = np.array(lst)

print("Vector created from a list:")
print(vctr)


# In[9]:


#Addition

import numpy as np

lst1 = [10,20,30,40,50]
lst2 = [1,2,3,4,5]

vctr1 = np.array(lst1)
vctr2 = np.array(lst2)

print("Vector created from a list 1:")
print(vctr1)

print("Vector created from a list 2:")
print(vctr2)

vctr_add = vctr1+vctr2

print("Addition of two vectors: ", vctr_add)


# In[10]:


# Substraction

import numpy as np

lst1 = [10,20,30,40,50]
lst2 = [1,2,3,4,5]

vctr1 = np.array(lst1)
vctr2 = np.array(lst2)

print("Vector created from a list 1:")
print(vctr1)

print("Vector created from a list 2:")
print(vctr2)

vctr_sub = vctr1-vctr2
print("Subtraction of two vectors: ", vctr_sub)


# In[11]:


#Multiplication

import numpy as np

lst1 = [10,20,30,40,50]
lst2 = [1,2,3,4,5]

vctr1 = np.array(lst1)
vctr2 = np.array(lst2)

print("Vector created from a list 1:")
print(vctr1)
print("Vector created from a list 2:")
print(vctr2)

vctr_mul = vctr1*vctr2

print("Multiplication of two vectors: ", vctr_mul)


# In[12]:


#Division

import numpy as np

lst1 = [10,20,30,40,50]
lst2 = [10,20,30,40,50]

vctr1 = np.array(lst1)
vctr2 = np.array(lst2)

print("Vector created from a list 1:")
print(vctr1)

print("Vector created from a list 2:")
print(vctr2)

vctr_div = vctr1/vctr2

print("Division of two vectors: ", vctr_div)


# # Matrices

# In[13]:


# Python program to create a matrix using numpy array 
import numpy as np

record = np.array([['Akmal', 89, 91], 
                  ['Hadhrami', 96, 82], 
                  ['Syazwan', 91, 81], 
                  ['Mazlan', 87, 91], 
                  ['Nooraini', 72, 79]])

matrix = np.reshape(record, (5,3))
print("The matrix is: \n", matrix)


# In[14]:


# create a 2D matrix
import numpy as np

matrix = np.matrix('3,4;5,6')
print(matrix)


# In[15]:


import numpy as np

record = np.array([['Akmal', 89, 91], 
                  ['Hadhrami', 96, 82], 
                  ['Syazwan', 91, 81], 
                  ['Mazlan', 87, 91], 
                  ['Nooraini', 72, 79]])

matrix = np.reshape(record, (5,3))

# Accessing record of Akmal
print(matrix[0])

# Accessing marks in the matrix subject of Mazlan
print("Mazlan's marks in AI subject: ", matrix[3][2])


# In[16]:


#Use dtype attribute of the array method to provide integers, float and even complex numbers in matrix
import numpy as np

array1 = np.array([[4,2,7,3], [2,8,5,2]])
print("Array of data type integers: \n", array1)

array2 = np.array([[1.5,2.2,3.1], [3,4.4,2]], dtype = "float")
print("Array of data type float: \n", array2)

array3 = np.array([[5,3,6], [2,5,7]], dtype = "complex")
print("Array of data type complex numbers: \n", array3)


# In[17]:


#Add two matrices and use the nested loop
matrix1 = [[23, 43, 12], 
          [43, 13, 55], 
          [23, 12, 13]]

matrix2 = [[4, 2, -1], 
          [5, 4, -34], 
          [0, -4, 3]]

matrix3 = [[0, 1, 0], 
          [1, 0, 0], 
          [0, 0, 1]]

matrix4 = [[0, 0, 0], 
          [0, 0, 0], 
          [0, 0, 0]]

matrices_length = len(matrix1)

for row in range(len(matrix1)):
    for column in range(len(matrix2[0])): 
        matrix4[row][column] = matrix1[row][column] + matrix2[row][column] + matrix3[row][column]
    
print("The sum of the matrices is =", matrix4)


# In[18]:


#Use @ for multiplication operator for matrix in Python
import numpy as np

matrix1 = np.matrix('3,4;5,6')
matrix2 = np.matrix('4,6;8,2')

print(matrix1 @ matrix2)


# In[19]:


#Matrix inverse
import numpy as np

A = np.matrix("3,4,6; 6,2,7; 6,4,6")

print(np.linalg.inv(A))


# In[20]:


#Matrix Transpose
import numpy as np

matrix = np.matrix('[5, 7, 6; 4, 2, 4]')

transpose = matrix.transpose()

print('Before transpose\n')
print(matrix)
print('\nAfter transpose\n')
print(transpose)


# # Tensors

# In[21]:


from numpy import array

T = array([
    [[1,2,3], [4,5,6], [7,8,9]], 
    [[11,12,13], [14,15,16], [17,18,19]], 
    [[21,22,23], [24,25,26], [27,28,29]], 
    ])

print(T.shape)
print(T)


# In[22]:


# tensor addition / subtraction / division

from numpy import array
A = array([
    [[1,2,3], [4,5,6], [7,8,9]], 
    [[11,12,13], [14,15,16], [17,18,19]], 
    [[21,22,23], [24,25,26], [27,28,29]], 
    ])
B = array([
    [[1,2,3], [4,5,6], [7,8,9]], 
    [[11,12,13], [14,15,16], [17,18,19]], 
    [[21,22,23], [24,25,26], [27,28,29]], 
    ])

Cadd = A + B
Csub = A - B 
Cdiv = A / B
print(Cadd)
print(Csub)
print(Cdiv)


# # PyTorch

# In[23]:


#Initializing tensors using NumPy Array and PyTorch tensors
import numpy as np
import torch

array = np.array([1, 2, 3])
print(f'NumPy Array: {array}')

tensor = torch.tensor([1,2,3])
print(f'PyTorch Tensor: {tensor}')


# In[24]:


#Initializing 2D arrays and 2D tensors using NumPy Array and PyTorch
array = np.array([[1,2,3],
                 [4,5,6]])
print(f'NumPy Array:\n{array}')

tensor = torch.tensor([[1,2,3], 
                      [4,5,6]])

print(f'\nPyTorch Tensor:\n{tensor}')


# In[31]:


import numpy as np
import torch

device = torch.device("cpu") #if normal gpu can use 'cpu'

#4D arrays
array1 = np.random.rand(100, 100, 100, 100)
array2 = np.random.rand(100, 100, 100, 100)

#4D tensors
tensor1 = torch.rand(100, 100, 100, 100).to(device)
tensor2 = torch.rand(100, 100, 100, 100).to(device)


# In[32]:


#used to measure the time execution
get_ipython().run_line_magic('%timeit', '')
np.matmul(array1, array2)


# In[33]:


#used to measure the time execution
get_ipython().run_line_magic('%timeit', '')
torch.matmul(tensor1, tensor2)

