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

w = random.rand(3)
print(w)
j=np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1],
])
errors = []
eta = 0.2
n = 100
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
for x, _ in training_data:
    result = dot(x, w)
    print("{}: -> {}".format(x[:2], unit_step(result)))
    l.append(unit_step(result))
    
for d, sample in enumerate(j):
    # Plot the negative samples
    if l[d] < 1:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)
z = []
k = []
b = w[2]
for x1 in range(-12,22):
#for x in training_data:
    x = float(x1/10)
    k.append(x)
    y = float((-(b/w[1])/(b/w[0]))*x - (b/w[1]))
    #print(w,x,y)
    z.append(y)
plt.plot(k,z)
plt.grid(True)
plt.ylim(-1.5,2.5)
plt.xlim(-1.5,2.5)
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("OR Perceptron")
plt.show()
#plt.plot([-0.5,0.9],[0.9,-0.5])
#plt.plot([0,1.5],[1.5,0])
