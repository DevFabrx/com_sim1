# Imports
import numpy as np
import matplotlib.pyplot as plt
import math
import utils

# start time measurement
start_time = utils.tic()

# mathematical constants
pi = np.pi
e = np.e

# creating a seed for random generator
np.random.seed(19680801)

def inverse_transformation(s, number_of_elements):
    rand_array = np.zeros(number_of_elements)
    epsilon = np.random.uniform(-10, 10, number_of_elements)
    for i in range(number_of_elements):
        rand_array[i] = pi * s * np.tan(epsilon[i])
    if rand_array.size == 1:
        return rand_array[0]
    else:
        return rand_array


def h_array(s, start, stop, steps):
    x = np.linspace(start, stop, num=steps)
    array = np.zeros(x.size)
    for i in range(x.size):
        array[i] = h(s, x[i])
    return array


def rho_array(s, start, stop, steps):
    a = np.linspace(start, stop, num=steps)
    array = np.zeros(a.size)
    for i in range(a.size):
        array[i] = rho(s, a[i])
    return array


def h(s, x):
    return 1/pi*(s/(math.pow(s, 2) + math.pow(x, 2)))


def g(s, x):
    return math.pow(np.sin(x), 2) * h(s, x)


def rho(s, x):
    z = (1 - math.pow(e, -2))/2
    return (1/(z*pi))*(math.pow(np.sin(x), 2)/(1+math.pow(x, 2)))


def rejection_method(s, number_of_elements):
    rand_array = np.zeros(number_of_elements)
    counter = 0
    while counter < number_of_elements:
        x_t = inverse_transformation(s, 1)
        r = np.random.uniform(-1, 1)
        if r * h(s, x_t) <= g(s, x_t):
            rand_array[counter] = x_t
            counter += 1
    return rand_array

'''

def create_hist(data, xlim, bins):

    return figure
'''

# Variables
s = 1
size = int(10e6)
bins = np.linspace(-10, 10, 10000).tolist()


# random numbers from the different generators
y_inverse = inverse_transformation(s, size)
y_rejection = rejection_method(s, size)


# histogram plots
plt.hist(y_inverse, bins=bins)
plt.show()

plt.hist(y_rejection, bins=bins)
plt.show()

# function plot
x = np.linspace(-10, 10, size)
y1 = h_array(s, -10, 10, size)
y1 = y1
# y1 = [h(s,i) for i in x]
y2 = rho_array(s, -10, 10, size)
y2 = y2
# y2 = [rho(s,i) for i in x]



# plot h function
plt.plot(x, y1)
plt.show()

# plot rho function
plt.plot(x, y2)
plt.show()

# output execution time5
utils.toc(start_time)