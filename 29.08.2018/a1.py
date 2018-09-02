from random import choice
import numpy as np
from numpy import array,random,dot
import matplotlib.pyplot as plt

unit_step = lambda x: 0 if x < 0 else 1

training_data = [
    (array([0,0,1]), 0),
    (array([0,1,1]), 1),
    (array([1,0,1]), 1),
    (array([1,1,1]), 1),
]

j=np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1],
])

errors = []
eta = 0.2
w = random.rand(3)
print(w)
n = 40
for i in range(n):
    x, expected = choice(training_data)
    result = dot(w, x)
    error = expected - unit_step(result)
    errors.append(error)
    w += eta * error * x    
    
print("\n")
print("respective weights:",w) 
print("\n") 
l = []  
for x, y in training_data:
    result = dot(x, w)
    print("{} -> {}".format(x[:2], unit_step(result)))
    l.append(unit_step(result))

for d, sample in enumerate(j):
    if l[d] < 1:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)
y = []
x = []
b = w[2]
for x1 in range(-12,22):
    x2 = float(x1/10)
    x.append(x2)
    y2 = float((-(b/w[1])/(b/w[0]))*x2 - (b/w[1]))
    y.append(y2)
    
plt.plot(x,y)
plt.grid(True)
plt.ylim(-1.5,2.5)
plt.xlim(-1.5,2.5)
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("OR Perceptron")
plt.show()
