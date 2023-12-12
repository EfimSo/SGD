#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 22:26:26 2023

@author: minhthubui
"""
import numpy as np
from autograd import grad
import autograd.numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sgd_robust_regression import (_robust_loss, make_sgd_robust_loss, generate_data, run_SGD, plot_iterates_and_squared_errors)

seed = 42   # seed for pseudorandom number generator
N = 100    # number of obervations
D = 10       # features dimensionality 



batchsize = 10       # batch size
init_param = np.zeros(D+1)  

init_stepsize = 0.2
stepsize_node = 5
k = 0.5
alpha = 0.51
stepsize_decayrate = 0
stepsize_decayrate_3 = alpha
nu = 5
true_beta, Y, Z = generate_data(N, D, seed)
sgd_loss, grad_sgd_loss = make_sgd_robust_loss(Y, Z, nu)
opt_param = np.concatenate(([1.0], true_beta))  # Optimal parameters

def run_SGD(grad_loss, epochs, init_param, init_stepsize, stepsize_decayrate, batchsize, N):
    K = (epochs * N) // batchsize # Count the number of iterations so N //batchsize is the number of iterations per epoch
    D = len(init_param)
    paramiters = np.zeros((K+1,D))
    paramiters[0] = init_param
    for k in range(K):
        inds = np.random.choice(N, batchsize)
        stepsize = init_stepsize / (k+1)**stepsize_decayrate
        paramiters[k+1] = paramiters[k] - stepsize*grad_loss(paramiters[k], inds) # average iterates here
    return paramiters

################################################
## (A) The effect of the number of iterations:##
################################################
epochs_list = [50, 100, 200, 400, 1000]    
def plot_epoch_and_squared_error(paramiters_last_iterate, paramiters_iterate_averaged, paramiters_last_iterate_decreasing, true_beta, opt_param, skip_epochs, epochs, N, batchsize, include_psi):
    skip_epochs = 0
    skip_iters = int(skip_epochs*N//batchsize)
    xs = np.linspace(skip_epochs, epochs, paramiters_last_iterate.shape[0] - skip_iters)
    plt.plot(xs,np.linalg.norm(paramiters_last_iterate - opt_param[np.newaxis,:], axis=1)**2, label = "Last Iterate")
    plt.plot(xs,np.linalg.norm(paramiters_iterate_averaged - opt_param[np.newaxis,:], axis=1)**2, label = "Iterate Averaged")
    plt.plot(xs,np.linalg.norm(paramiters_last_iterate_decreasing - opt_param[np.newaxis,:], axis=1)**2, label = "Iterate Decreasing")
    plt.xlabel('epoch')
    plt.ylabel(r'$\|x_k - x_{\star}\|_2^2$')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.title("Increasing Epochs and Squared Error")
    sns.despine()
    plt.show()

def plot_K_and_squared_error(paramiters_last_iterate, paramiters_iterate_averaged, paramiters_last_iterate_decreasing, true_beta, opt_param, skip_epochs, epochs, N, batchsize, include_psi):
    K = (epochs * N) // batchsize
    skip_iters = int(skip_epochs*N//batchsize)
    xs = np.linspace(0, 10000, paramiters_last_iterate.shape[0] - skip_iters)
    plt.plot(xs,np.linalg.norm(paramiters_last_iterate - opt_param[np.newaxis,:], axis=1)**2, label = "Last Iterate")
    plt.plot(xs,np.linalg.norm(paramiters_iterate_averaged - opt_param[np.newaxis,:], axis=1)**2, label = "Iterate Averaged")
    plt.plot(xs,np.linalg.norm(paramiters_last_iterate_decreasing - opt_param[np.newaxis,:], axis=1)**2, label = "Iterate Decreasing")
    plt.xlabel('Number of iterations')
    plt.ylabel(r'$\|x_k - x_{\star}\|_2^2$')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.title("Number of Iterations and Squared Error")
    sns.despine()
    plt.show()

for epochs in epochs_list:
    paramiters_last_iterate = run_SGD(grad_sgd_loss, epochs, init_param, init_stepsize, stepsize_decayrate, batchsize, N)
    paramiters_iterate_averaged = np.array([np.mean(paramiters_last_iterate[i//2:i+1], axis=0) for i in range(paramiters_last_iterate.shape[0])])
    paramiters_last_iterate_decreasing = run_SGD(grad_sgd_loss, epochs, init_param, stepsize_node, stepsize_decayrate_3, batchsize, N)

plot_epoch_and_squared_error(paramiters_last_iterate, paramiters_iterate_averaged, paramiters_last_iterate_decreasing, true_beta, opt_param, skip_epochs = 0, epochs=epochs, N=N, batchsize=batchsize, include_psi=True)
plot_K_and_squared_error(paramiters_last_iterate, paramiters_iterate_averaged, paramiters_last_iterate_decreasing, true_beta, opt_param, skip_epochs = 0, epochs=epochs, N=N, batchsize=batchsize, include_psi=True)

