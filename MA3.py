""" MA3.py

Student:Thilia Lindh Seger
Mail: thilia.lindh-seger@3653.uu.se
Reviewed by:
Date reviewed:

"""
import random
import numpy as np
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
from numba import njit

# Exc1
def approximate_pi(n):
    inside = 0
    x_outside = []
    y_outside = []
    x_inside = []
    y_inside = []


    for i in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        circ = x**2 + y**2

        if circ <= 1:
            inside += 1
            x_inside.append(x)
            y_inside.append(y)
        else: 
            x_outside.append(x)
            y_outside.append(y)

    pi = 4 * (inside/n)

    plt.scatter(x_inside, y_inside, color= 'red', label='Inside circle')
    plt.scatter(x_outside, y_outside, color= 'blue', label='Outside circle')
    plt.title(f'Approximation of π with {n} random dots')
    plt.legend()
    plt.axis('equal')
    plt.savefig(f'pi_{n}.png')
    plt.show()

    #print(f'When n={n}, π is approximately: {pi}')
    return pi


# Exc2, approximation
def sphere_volume(n, d): 
    inside = 0

    for i in range(n):
        point = [random.uniform(-1, 1) for j in range(d)] #list comprehension
        
        x_sqrt = list(map(lambda x:x**2, point)) #map(function, list)
        x_sqrt = sum(x_sqrt)
        if x_sqrt <= 1:
            inside += 1

    V = 2**d * (inside/n) 
    return V

#Exc2, real value
def hypersphere_exact(n, d):
    V = (np.pi**(d/2))/m.gamma(d/2 + 1)
    
    #print(f'Approx value when (n, d) = ({n}, {d}) is: {sphere_volume(n, d)}')
    #print(f'Real value when (n, d) = ({n}, {d}) is: {V}')

    return V

#Exc3: numba version

# time1: 8.7034 sec
# approx value1: 1.890304
# time2: 12.0061 sec
# approx value2: 1.961984
# time3: 12.2766 sec
# approx value3: 1.775616

# numba time1: 3.0165 sec
# approx value1: 1.888256
# numba time2: 0.8688 sec
# approx value2: 1.78176
# numba time3: 0.8715 sec
# approx value3: 1.867776
@njit
def sphere_volume_numba(n:int, d:int)->float:
    inside = 0
    for i in range(n):
        x_sum = 0
        point = [random.uniform(-1, 1) for j in range(d)]
            
        for x in point:
            x_sqrt = x**2
            x_sum += x_sqrt

        if x_sum <= 1:
            inside += 1
    
    V = 2**d * (inside/n) 
    return V

#Exc4: parallel code - parallelize actual computations by splitting data
# time1: 8.4699 sec
# volume1: 2.001
# time2: 7.6559 sec
# volume2: 1.804
# time3: 7.6152 sec
# volume3: 1.964



def sphere_volume_parallel(n, d, np=10):
    n_np = n // np
    tasks = []
    V = []
    with future.ProcessPoolExecutor() as ex:
        for i in range(np):
            task = ex.submit(sphere_volume, n_np, d)
            tasks.append(task)

        for task in tasks:
            V.append(task.result())

    return mean(V)
    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # Exc3
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc3: Sequential time of {d} and {n}: {stop-start}")
    print("What is numba time?")

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")
    print("What is parallel time?")

    
    

if __name__ == '__main__':
	
    n = 1_000_000
    d = 11

    start1 = pc()
    V = sphere_volume_parallel(n, d)
    stop1 = pc()

    start2 = pc()
    V = sphere_volume(n, d)
    stop2 = pc()

    print(f"Parallel volume: {V}")
    print(f"Parallel time: {stop1-start1} sec")
    print(f"Sequential time: {stop2-start2} sec")

    
    main()
    