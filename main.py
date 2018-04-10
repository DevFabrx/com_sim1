import numpy as np
import matplotlib.pyplot as plt
import math



def inverse_transformation(s, number_of_elements):
    rand_array = []
    epsilon = np.random.uniform(-1, 1, number_of_elements)
    for i in range(number_of_elements):
        rand_array.append(np.pi * s * np.tan(epsilon[i]))
    if len(rand_array) == 1:
        return rand_array[0]
    else:
        return rand_array


def h(s, x):
    return 1/np.pi*(s/(math.pow(s, 2) + math.pow(x, 2)))


def g(s, x):
    return math.pow(np.sin(x), 2) * h(s, x)

def rho(s, x):
    return 1/np.pi*(math.pow(np.sin(x), 2)/(1+math.pow(x, 2)))


def rejection_method(s, number_of_elements):
    rand_array = []
    counter = 0
    while counter < number_of_elements:
        x_t = inverse_transformation(s, 1)
        r = np.random.uniform(-1, 1)
        if r*h(s, x_t) < g(s, x_t):
            rand_array.append(x_t)
            counter += 1
    return rand_array


# Variables
s = 0.1
size = 100000
bins = np.array[np.linspace(-10, 10, 1)]

# random numbers from the different generators
y_inverse = inverse_transformation(s, size)
y_rejection = rejection_method(s, size)

# histogram plots
plt.hist(y_inverse, bins=100)
plt.xlim(bins)
plt.show()

plt.hist(y_rejection, bins=100)
plt.xlim(bins)
plt.show()


y = [rho(1, i) for i in range(size)]
x = np.linspace(-10, 10, size)

plt.plot(x,y)
plt.show()
