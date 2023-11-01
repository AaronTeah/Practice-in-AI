#!/usr/bin/env python
# coding: utf-8

# In[1]:


who = 'Python'
what = 'cool'
str1 = f"{who} is {what}!" 
print(str1)


# In[2]:


str2 = "{} is {}".format(who, what)
print(str2)


# In[3]:


str3 = "%s is %s"%(who, what)
print(str3)


# In[4]:


string = 'The sun shines bright'
print(string.title())
print(string.split())
print(string.count('s', 0, len(string)))


# In[5]:


string2 = 'I am handsome!'
print("handsome" in string2)


# In[6]:


x = input("x: ")
print(type(x))
y = float(x) + 1
print(f"x:{x} y:{y}")


# In[7]:


name = input("Name: ")
matrixNumber = input("Matrix Number: ")
birthday = input("Year of Birthday: ")
age = 2023 - int(birthday)
print(f"""Name:{name}
Matrix Number:{matrixNumber}
Birthday:{birthday}
Age:{age}""")


# In[8]:


5 == 5


# In[9]:


5 == 6


# In[10]:


print(type(True))


# In[11]:


10 == '10'


# In[13]:


x = 10
if x >= 10: 
    print('x is positive value')


# In[14]:


if x % 2 == 0: 
    print('x is even')
else: 
    print("x is odd")


# In[15]:


x = 10
y = 11
if x < y: 
    print("x is less than y")
elif x > y:
    print("x is greater than y")
else: 
    print("x and y are equal")


# In[18]:


age = 22
message = "Eligible" if age >= 18 else "Not eligible"
print(message)


# In[20]:


high_income = True
good_credit = True
student = True

if high_income and good_credit: 
    print("Eligible")
else:
    print("Not eligible")

if high_income and good_credit and not student:
    print("Eligible")
else: 
    print("Not Eligible")


# In[23]:


for n in range(3): 
    print("sending a message")
    
for n in range(3): 
    print("Attempt", n)


# In[24]:


for number in range(3):
    print("Attempt", number + 1, (number+1) * '.')


# In[25]:


for x in range(2): 
    for y in range(2): 
        print(f"{x} {y}")


# In[1]:


def greet(): 
    print("Hello there!")
    print("Welcome to Function in Python")

greet()


# In[5]:


def get_greeting(name):
    return f"Hi {name}"

message = get_greeting("Aaron")
print(message)

