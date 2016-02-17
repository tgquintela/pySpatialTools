
"""
"""

import numpy as np

import multiprocessing
from multiprocessing import Queue
import time


def distribute_tasks(n, memlim):
    """Util function for distribute tasks in matrix computation in order to
    save memory or to parallelize computations."""
    lims = []
    inflim = 0
    while True:
        if inflim + memlim >= n:
            lims.append([inflim, n])
            break
        else:
            lims.append([inflim, inflim+memlim])
            inflim = inflim+memlim
    return lims


def reshape_limits(lims, limits):
    new_limits = []
    xs = np.linspace(limits[0], limits[1], lims[-1][1]+1)
    for i in range(len(lims)):
        new_limits.append([xs[lims[i][0]], xs[lims[i][1]]])
    return new_limits

#import threading
#import os
#import queue
#import redis
#import rq
#
#
#def parallel_process(f, args_l):
#    return_values = Queue()
#    jobs = []
#    for i in range(len(args_l)):
#        p = multiprocessing.Process(target=some_long_process, args=(i, i**2, return_values))
#        jobs.append(p)
#        p.start()
#
#return_values = Queue()
#some_long_process = lambda x, y, q: q.put(x**y)
#
#for i in range(8):
#    p = multiprocessing.Process(target=some_long_process, args=(i, i**2, return_values))
#    jobs.append(p)
#    p.start()
#time.sleep(0.5)
#for i in range(8):
#    print(return_values.get())
#
