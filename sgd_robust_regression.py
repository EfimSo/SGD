from autograd import grad
import autograd.numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def _robust_loss(psi, beta, nu, Y, Z):
    scaled_sq_errors = np.exp(-2*psi)  * (np.dot(Z, beta) - Y)**2
    if nu == np.inf:
        return scaled_sq_errors/2 + psi
    return (nu + 1)/2 * np.log(1 + scaled_sq_errors / nu) + psi


def make_sgd_robust_loss(Y, Z, nu):
    N = Y.size
    sgd_loss = lambda param, inds: np.mean(_robust_loss(param[0], param[1:], nu, Y[inds], Z[inds])) + np.sum(param**2)/(2*N)
    grad_sgd_loss = grad(sgd_loss)
    return sgd_loss, grad_sgd_loss


def generate_data(N, D, seed):
    rng = np.random.default_rng(seed)
    # generate multivariate t covariates with 10 degrees
    # of freedom and non-diagonal covariance 
    t_dof = 10
    locs = np.arange(D).reshape((D,1))
    cov = (t_dof - 2) / t_dof * np.exp(-(locs - locs.T)**2/4)
    Z = rng.multivariate_normal(np.zeros(D), cov, size=N)
    Z *= np.sqrt(t_dof / rng.chisquare(t_dof, size=(N, 1)))
    # generate responses using regression coefficients beta = (1, 2, ..., D)
    # and t-distributed noise 
    true_beta = np.arange(1, D+1)
    Y = Z.dot(true_beta) + rng.standard_t(t_dof, size=N)
    # for simplicity, center responses 
    Y = Y - np.mean(Y)
    return true_beta, Y, Z


def run_SGD(grad_loss, epochs, init_param, init_stepsize, stepsize_decayrate, batchsize, N):
    K = (epochs * N) // batchsize # num iterations
    D = init_param.size
    paramiters = np.zeros((K+1,D))
    paramiters[0] = init_param
    for k in range(K):
        inds = np.random.choice(N, batchsize)
        stepsize = init_stepsize / (k+1)**stepsize_decayrate
        paramiters[k+1] = paramiters[k] - stepsize*grad_loss(paramiters[k], inds)
    return paramiters


def plot_iterates_and_squared_errors(paramiters, true_beta, opt_param, skip_epochs, epochs, N, batchsize, include_psi=True):
    D = true_beta.size
    param_names = [r'$\beta_{{{}}}$'.format(i) for i in range(D)]
    if include_psi:
        param_names = [r'$\psi$'] + param_names
    else:
        paramiters = paramiters[:,1:]
        opt_param = opt_param[1:]
    skip_epochs = 0
    skip_iters = int(skip_epochs*N//batchsize)
    xs = np.linspace(skip_epochs, epochs, paramiters.shape[0] - skip_iters)
    plt.plot(xs, paramiters[skip_iters:]);
    plt.plot(np.array(D*[[xs[0], xs[-1]]]).T, np.array([true_beta,true_beta]), ':')
    plt.xlabel('epoch')
    plt.ylabel('parameter value')
    plt.legend(param_names, bbox_to_anchor=(0,1.02,1,0.2), loc='lower left',
               mode='expand', borderaxespad=0, ncol=4, frameon=False)
    sns.despine()
    plt.show()
    plt.plot(xs, np.linalg.norm(paramiters - opt_param[np.newaxis,:], axis=1)**2)
    plt.xlabel('epoch')
    plt.ylabel(r'$\|x_k - x_{\star}\|_2^2$')
    plt.yscale('log')
    sns.despine()
    plt.show()


seed = 1234
data = generate_data(10000, 10, seed)
