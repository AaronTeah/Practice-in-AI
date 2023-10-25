#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('Hello, World')


# # Welcome to new Jupyter Notebook
# - This is my first tutorial using Jupyter
# - The language is Python
# :)

# In[3]:


age = 22
print(age)


# # Tutorial 3
# TASK 1 – Python Basic
# In this task, you will learn the basics of Python syntax by running some sample code. Python is known for its simplicity and readability, making it an ideal language for beginners.

# In[8]:


# Primitive Data Types
my_integer = 42  # Integer data type
my_float = 3.141592  # Float data type
my_string = "Hello, Python!"  # String data type
my_boolean = True  # Boolean data type


# In[10]:


# Container Data Types
my_list = [1, 2, 3, 4, 5]  # List data type (a collection of values)
my_dict = {'name': 'John', 'age': 30}  # Dictionary data type (key-value pairs)


# In[11]:


#Display output
print("Integer:", my_integer)
print("Float:", my_float)
print("String:", my_string)
print("Boolean:", my_boolean)
print("List:", my_list)
print("Dictionary:", my_dict)


# # TASK 2 - Writing a BMI Calculator in Python-
# In this task, you will apply the knowledge you've gained about data types, assignment statements, and arithmetic operations to create a simple Python program for calculating the Body Mass Index (BMI). There is no need for user input; instead, you will directly assign values to variables and run the code to obtain the BMI. Use the (BMI) formula given below. 
# 

# In[19]:


name = 'Azanizam'
weight_kg = 70.0
height_m = 1.75
bmi = weight_kg / height_m ** 2
print('BMI for ' + name)
print(f'My weight = {weight_kg} kg')
print(f'My height = {height_m} m')
print(f'My BMI = {round(bmi, 2)} ')

