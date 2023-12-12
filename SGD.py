# import numpy as np
from autograd import grad
import autograd.numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sgd_robust_regression import (make_sgd_robust_loss, generate_data)

seed = 42   # seed for pseudorandom number generator
N = 10000    # number of obervations
D = 10       # features dimensionality 

batchsize = 10       # batch size
init_param = np.zeros(D+1)  

init_stepsize = 0.2
stepsize_node = 5
k = 0.5
alpha = 0.51
stepsize_decayrate = 0
stepsize_decayrate_alpha = alpha
nu = 5
true_beta, Y, Z = generate_data(N, D, seed)
sgd_loss, grad_sgd_loss = make_sgd_robust_loss(Y, Z, nu)
opt_param = np.concatenate(([1.0], true_beta))  # Optimal parameters

# Implements Iterate averaging after start_averaging epochs
def run_SGD(grad_loss, epochs, init_param, init_stepsize, stepsize_decayrate, batchsize, N, start_averaging=0):
    K = (epochs * N) // batchsize # num iterations
    D = init_param.size
    paramiters = np.zeros((K+1,D))
    averages = np.zeros(((K+1),D))
    paramiters[0] = init_param
    sum, avg = np.zeros_like(init_param), np.zeros_like(init_param)
    s = (start_averaging * N) // batchsize # number of iterations when to start running average
    for k in range(K):
        inds = np.random.choice(N, batchsize)
        stepsize = init_stepsize / (k+1)**stepsize_decayrate
        paramiters[k+1] = paramiters[k] - stepsize*grad_loss(paramiters[k], inds)
        ind = paramiters.shape[0] // 2
        avg = np.sum(paramiters[ind:k+1])
        # sum = sum + paramiters[k+1]
        # avg = sum / (k+1)
        averages[k+1] = avg
    return paramiters, averages


# Plots regression parameters (if plot_first = True) and error rates 
# discards the averages before start_avg epochs
def plot_iterates_and_squared_errors(paramiters, paramiters_averaged, true_beta, opt_param, skip_epochs, epochs, N, batchsize, include_psi=True, start_avg = 0, plot_first = True):
    D = true_beta.size
    param_names = [r'$\beta_{{{}}}$'.format(i) for i in range(D)]
    if include_psi:
        param_names = [r'$\psi$'] + param_names
    else:
        paramiters = paramiters[:,1:]
        opt_param = opt_param[1:]
    skip_epochs = 0
    skip_iters = int(skip_epochs*N//batchsize)
    s = (start_avg * N)  // batchsize       # number iterations when to start running average 
    xs = np.linspace(skip_epochs, epochs, paramiters.shape[0] - skip_iters)
    avs = np.linspace(start_avg, epochs, paramiters.shape[0] - s)
    if plot_first:
        plt.plot(xs, paramiters[skip_iters:])
        plt.plot(avs, paramiters_averaged[max(skip_iters, s):])
        plt.plot(np.array(D*[[xs[0], xs[-1]]]).T, np.array([true_beta,true_beta]), ':')
        plt.xlabel('epoch')
        plt.ylabel('parameter value')
        plt.legend(param_names, bbox_to_anchor=(0,1.02,1,0.2), loc='lower left',
                mode='expand', borderaxespad=0, ncol=4, frameon=False)
        sns.despine()
        plt.show()
    plt.plot(xs, np.linalg.norm(paramiters - opt_param[np.newaxis,:], axis=1)**2)
    plt.plot(avs, np.linalg.norm(paramiters_averaged[s:, :] - opt_param[np.newaxis:], axis=1)**2)
    plt.xlabel('epoch')
    plt.ylabel(r'$\|x_k - x_{\star}\|_2^2$')
    plt.yscale('log')
    sns.despine()
    plt.show()

# A) Iterations
# Run SGD with iterate averaging starting at 0 epochs.
epochs = 150
paramiters, paramiters_averaged = run_SGD(grad_sgd_loss, epochs, init_param, init_stepsize, stepsize_decayrate, batchsize, N)
_ = plot_iterates_and_squared_errors(paramiters, paramiters_averaged, true_beta, opt_param, 0, epochs, N, batchsize)

# Run SGD with iterate averaging starting at $\lfloor k / 2 \rfloor$.
paramiters_half, paramiters_averaged_half = run_SGD(grad_sgd_loss, epochs, init_param, init_stepsize, stepsize_decayrate, batchsize, N, start_averaging=epochs // 2)
_ = plot_iterates_and_squared_errors(paramiters_half, paramiters_averaged_half, true_beta, opt_param, 0, epochs, N, batchsize, start_avg=epochs // 2)

# Error analysis
print(f"Last iterate error no iterate averaging = {np.linalg.norm(paramiters[-1, :] - opt_param)}")
print(f"Last iterate error iterate averaging starting at 0 = {np.linalg.norm(paramiters_averaged[-1, :] - opt_param)}")
print(f"Last iterate error iterate averaging starting at epochs // 2 = {np.linalg.norm(paramiters_averaged_half[-1, :] - opt_param)}\n")
def spread_iters(params, eps, label):
    norms = np.linalg.norm(params[(eps * N) // batchsize:, :] - opt_param[np.newaxis,:], axis=1)
    print(f"Max{label} iterate error after {eps} epochs {norms.max()}")
    print(f"Min iterate error after {eps} epochs{label} {norms.min()}")
    print(f"Mean{label} iterate error after {eps} epochs {norms.mean()}\n")
spread_iters(paramiters, 100, "")
spread_iters(paramiters_averaged, 100, " averaged")
print()
spread_iters(paramiters, 150, "")
spread_iters(paramiters_averaged, 150, " averaged")
spread_iters(paramiters_averaged_half, 150, " averaged at half")

# B) Effect of initialization 
x0 = np.random.uniform(0, D, D+1) # sample from uniform
print(np.linalg.norm(x0 - opt_param))
p, p_av = run_SGD(grad_sgd_loss, epochs, x0, init_stepsize, stepsize_decayrate, batchsize, N, start_averaging=0)
_ = plot_iterates_and_squared_errors(p, p_av, true_beta, opt_param, 0, epochs, N, batchsize, start_avg=0, plot_first=False)

# Same plot but with k // 2 averaging
epochs = 50
p, p_av = run_SGD(grad_sgd_loss, epochs, x0, init_stepsize, stepsize_decayrate, batchsize, N, start_averaging=epochs // 2)
_ = plot_iterates_and_squared_errors(p, p_av, true_beta, opt_param, 0, epochs, N, batchsize, start_avg=epochs // 2, plot_first=False), 

# C) Step size
stepsize = 0.8 # larger init step
p, p_av = run_SGD(grad_sgd_loss, epochs, init_param, stepsize, stepsize_decayrate, batchsize, N, start_averaging=epochs // 2)
_ = plot_iterates_and_squared_errors(p, p_av, true_beta, opt_param, 0, epochs, N, batchsize, start_avg=epochs // 2, plot_first=False), 

# Stepsize decay with small initial step size
epochs=500
p, p_av = run_SGD(grad_sgd_loss, epochs, init_param, init_stepsize, stepsize_decayrate_alpha, batchsize, N, start_averaging=epochs // 2)
_ = plot_iterates_and_squared_errors(p, p_av, true_beta, opt_param, 0, epochs, N, batchsize, start_avg=epochs // 2, plot_first=False), 
print(f"Last iterate error no iterate averaging = {np.linalg.norm(p[-1, :] - opt_param)}")
print(f"Last iterate error iterate averaging = {np.linalg.norm(p_av[-1, :] - opt_param)}")

# Stepsize decay and larger inital step size
epochs = 100
stepsize = 2.5
p, p_av = run_SGD(grad_sgd_loss, epochs, init_param, stepsize, stepsize_decayrate_alpha, batchsize, N, start_averaging=epochs // 2)
_ = plot_iterates_and_squared_errors(p, p_av, true_beta, opt_param, 0, epochs, N, batchsize, start_avg=epochs // 2, plot_first=False), 
print(f"Last iterate error no iterate averaging = {np.linalg.norm(p[-1, :] - opt_param)}")
print(f"Last iterate error iterate averaging = {np.linalg.norm(p_av[-1, :] - opt_param)}")

# D) Batch size
# Small B
epochs = 150
batchsize_d = batchsize // 10    # B = 1
p, p_av = run_SGD(grad_sgd_loss, epochs, init_param, init_stepsize, stepsize_decayrate, batchsize_d, N, start_averaging=epochs // 2)
_ = plot_iterates_and_squared_errors(p, p_av, true_beta, opt_param, 0, epochs, N, batchsize_d, start_avg=epochs // 2, plot_first=False), 
print(f"Last iterate error no iterate averaging = {np.linalg.norm(p[-1, :] - opt_param)}")
print(f"Last iterate error iterate averaging = {np.linalg.norm(p_av[-1, :] - opt_param)}")

# Large B
epochs = 400
batchsize_d = 100
p, p_av = run_SGD(grad_sgd_loss, epochs, init_param, init_stepsize, stepsize_decayrate, batchsize_d, N, start_averaging=epochs // 2)
_ = plot_iterates_and_squared_errors(p, p_av, true_beta, opt_param, 0, epochs, N, batchsize_d, start_avg=epochs // 2, plot_first=False), 
print(f"Last iterate error no iterate averaging = {np.linalg.norm(p[-1, :] - opt_param)}")
print(f"Last iterate error iterate averaging = {np.linalg.norm(p_av[-1, :] - opt_param)}")

# E) Loss choice
std_reg_loss, grad_std_reg_loss = make_sgd_robust_loss(Y, Z, np.inf) # regular regression loss (normal noise)
epochs = 150
p, p_av = run_SGD(grad_std_reg_loss, epochs, init_param, init_stepsize, stepsize_decayrate, batchsize, N, start_averaging=epochs // 2)
_ = plot_iterates_and_squared_errors(p, p_av, true_beta, opt_param, 0, epochs, N, batchsize, start_avg=epochs // 2, plot_first=False)
print(f"Last iterate error no iterate averaging = {np.linalg.norm(p[-1, :] - opt_param)}")
print(f"Last iterate error iterate averaging = {np.linalg.norm(p_av[-1, :] - opt_param)}")


