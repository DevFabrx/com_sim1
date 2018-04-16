# Imports
import numpy as np
import matplotlib.pyplot as plt
import math
import utils
import logging

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)


# start time measurement
start_time = utils.tic()

# mathematical constants
pi = np.pi
e = np.e

# Variables
s = 1
size = 10**5
xmin = -10
xmax = math.fabs(xmin)
nr_bins = 150
bins = np.linspace(xmin, xmax, nr_bins)
width = math.fabs(bins[1]-bins[0])

# function data
x = np.linspace(xmin, xmax, size)


# creating a seed for random generator
np.random.seed(19680801)


def h_inv(s, x):
    return s * math.tan(pi*(x-0.5))

def h(s, x):
    return (1/pi)*(s/(s**2 + x**2))


def g(s, x):
    return math.sin(x)**2 * h(x, s)


def rho(x):
    z = (1 - e**-2)/2
    return (1/(z*pi))*(math.sin(x)**2)/(1+x**2)


def uniform(a, b):
    return 1/(b-a)

def inverse_transformation(s, number_of_elements):
    rand_array = np.zeros(number_of_elements)
    epsilon = np.random.uniform(0, 1, number_of_elements)
    for index in range(number_of_elements):
        # rand_array[index] = pi * s * np.tan(epsilon[index])
        rand_array[index] = h_inv(s, epsilon[index])
    if rand_array.size == 1:
        return rand_array[0]
    else:
        return rand_array


def inverse_transformation2(a, b, number_of_elements):
    rand_array = np.zeros(number_of_elements)
    epsilon = np.random.uniform(xmin, xmax, number_of_elements)
    for index in range(number_of_elements):
        rand_array[index] = epsilon[index]*(b - a)
    if rand_array.size == 1:
        return rand_array[0]
    else:
        return rand_array


def rejection_method(s_value, number_of_elements):
    rand_array = np.zeros(number_of_elements)
    ac_counter = 0
    counter = 0
    while counter < number_of_elements:
        x_inverse = inverse_transformation(s_value, 1)
        r = np.random.rand()
        if r * 3 * h(s_value, x_inverse) < rho(x_inverse):
            rand_array[counter] = x_inverse
            counter += 1
        ac_counter += 1
    acceptance_rate = number_of_elements / ac_counter * 100
    return rand_array, acceptance_rate


def rejection_method2(number_of_elements):
    rand_array = np.zeros(number_of_elements)
    ac_counter = 0
    counter = 0
    while counter < number_of_elements:
        x_inverse = inverse_transformation2(0, 1, 1)
        r = np.random.rand()
        c = 0.3*(xmax-xmin)
        if r * c * uniform(xmin, xmax) < rho(x_inverse):
            rand_array[counter] = x_inverse
            counter += 1
        ac_counter += 1
    acceptance_rate = number_of_elements / ac_counter * 100
    return rand_array, acceptance_rate





# random numbers from the different generators
y_inverse = inverse_transformation(s, size)
y_rejection, acceptance_rate = rejection_method(s, size)
y_inverse2 = inverse_transformation2(0, xmax, size)
y_rejection2, acceptance_rate2 = rejection_method2(size)


# Calculate functions for plotting
h_func = [h(s, i) for i in x]
g_fun = [g(s, i) for i in x]
rho_func = [rho(i) for i in x]
h_func_rejection = [3 * h(s, i) for i in x]
const_function = [uniform(xmin, xmax) for i in x]
const_function_scaled = [0.3*(xmax-xmin)*uniform(xmin, xmax) for i in x]


# Plot h(s, x), rho(x) and constant function
fig_functions = plt.figure(0)
plt.plot(x, h_func)
plt.plot(x, rho_func)
plt.plot(x, const_function)
plt.legend(["h(s,x)", "rho(x)", "1/(b-a)"])
plt.title("Functions")
plt.xlabel("x")
plt.ylabel("Probability")
plt.show()


# Plot inverse method
fig_inverse = plt.figure(1)
plt.hist(y_inverse, bins, normed=1)
plt.plot(x, h_func, 'r')
plt.xlabel("x")
plt.ylabel("Probability")
plt.title("Inverse Method for a Cauchy distribution")
plt.legend(["h(x,s)", "Histogram data (inverse)"])
plt.show()


# Plot rejection method
fig_rejection = plt.figure(2)
plt.hist(y_rejection, bins, normed=1)
plt.plot(x, rho_func, 'r')
plt.plot(x, h_func_rejection, 'g-.')
plt.ylim([0, 0.4])
plt.xlabel("x")
plt.ylabel("Probability")
plt.title("Rejection Method (Cauchy Envelope)")
plt.legend(["rho(x)", "3*h(s,x)", "Histogram data (rejection)", ])
plt.show()


# Plot rejection method 2
fig_rejection2 = plt.figure(3)
plt.hist(y_rejection2, bins, normed=1)
plt.plot(x, rho_func, 'r')
plt.plot(x, const_function_scaled, 'g-.')
plt.ylim([0, 0.4])
plt.xlabel("x")
plt.ylabel("Probability")
plt.title("Rejection Method (Constant Function)")
plt.legend(["rho(x)", "f(x)=3/10", "Histogram data"])
plt.show()


plt.figure(4)
plt.hist(y_inverse2, bins, normed=1)
plt.plot(x, const_function)
plt.show()

area_cauchy_exact = 3 * 2 * math.atan(xmax) / pi
area_rho_exact = 0.921348
area_constant_exact = 0.3 * 20

# Acceptance Rates
print("Acceptance rate cauchy exact: {:.2f}%".format(area_rho_exact / area_cauchy_exact * 100))
print("Acceptance rate rejection method (cauchy): {:.2f}%\n".format(acceptance_rate))

print("Acceptance rate constant function exact: {:.2f}%".format(area_rho_exact / area_constant_exact * 100))
print("Acceptance rate rejection method (constant): {:.2f}%\n".format(acceptance_rate2))

# Save plots to pdf file
utils.save_fig_to_pdf("ComSim1.pdf", fig_functions, fig_inverse, fig_rejection, fig_rejection2)


# Measure total program runtime
utils.toc(start_time)