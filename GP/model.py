import gpytorch.constraints
import torch
import torch.nn as nn
import gpytorch
import torch.utils
import torch.utils.data
import copy
from torch.func import stack_module_state, functional_call
from typing import List
from torch import Tensor
import math



class PauliProductKernel(gpytorch.kernels.Kernel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Lower-triangular Cholesky factor of C
        self.raw_L = torch.nn.Parameter(
            torch.eye(4)
        )

    def forward(
        self,
        x1,
        x2,
        diag=False,
        last_dim_is_batch=False,
        **params,
    ):

        # C = L L^T
        L = torch.tril(self.raw_L)

        # Make diagonal positive
        diag_L = torch.diagonal(L)
        positive_diag = torch.nn.functional.softplus(diag_L)

        L = L - torch.diag(diag_L) + torch.diag(positive_diag)

        C = L @ L.T

        # Normalize to correlation matrix
        d = torch.sqrt(torch.diag(C))
        C = C / d[:, None] / d[None, :]

        # x1: N x n
        # x2: M x n

        # C[x1, x2] gives N x M x n
        per_dimension = C[x1[:, None, :], x2[None, :, :]]

        # Product over Pauli-string positions
        K = per_dimension.prod(dim=-1)

        if diag:
            return K.diagonal()

        return K

# vectorized version of FullLocalNW
# train_inputs: X of training set
# train_targets: Y of training set
# likelihood: Likelihood object from GPytorch
# kernel_cls: class name of Kernel
# target_dim: Number of "parallel" GPs (i.e. number of targets for same x)
class ShadowGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood, kernel_cls, 
                 kernel_parameters={}):
        super().__init__(train_inputs, train_targets, likelihood)
        kernel_parameters = dict(**kernel_parameters)


        self.mean_module = gpytorch.means.ZeroMean()
        #self.decay_rates = nn.Parameter(init_decay_rate*torch.ones((self.target_dim,)))
        self.kernel = kernel_cls(**kernel_parameters)
    
    def forward(self, x):
        mean_x = self.mean_module(x) 
        covar_x = self.kernel(x) 
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) 


# this function gives the respecive object to the training loop
def init_GP_model(train_loader: torch.utils.data.DataLoader, kernel_type, 
                  kernel_parameters: dict):

    train_x, train_y = [], []
    for x, y in train_loader:
        train_x.append(x)
        train_y.append(y)

    train_x = torch.cat(train_x)
    train_y = torch.cat(train_y).permute(1,0)

    #likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(0.005,1),
    #                                                     batch_shape=torch.Size([target_dim]))
    #print(train_x.shape)
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.zeros((train_x.shape[2], train_x.shape[2])))
    

    if kernel_type == "PauliProduct":
        kernel_cls = PauliProductKernel

    #model = FullLocalGP(train_x, train_y, likelihood, kernel_cls, 
                        #kernel_parameters, geometry_parameters)

    # TODO: Swap this with 
    model = ShadowGPModel(train_x, train_y, likelihood,
                          kernel_cls, kernel_parameters)
    
    return model, train_x, train_y, likelihood
