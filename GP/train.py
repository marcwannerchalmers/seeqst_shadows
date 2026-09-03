# Here comes the offline GP learner
        
#from learner.learner import init_learner
import torch
import os
import gpytorch
import pandas as pd
from torch import nn


class MLLLoss(nn.Module):
    def __init__(self, model: gpytorch.models.GP, 
                 likelihood: gpytorch.likelihoods.Likelihood):
        super().__init__()
        self.model = model
        self.likelihood = likelihood
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def forward(self, inputs, targets):
        #print("modx, ", self.mll(inputs, targets).shape)
        return -torch.mean(self.mll(inputs, targets), dim=0)

def train_gp(cfg, 
             model, 
             likelihood,
             train_x,
             train_y,
             loss_cls=MLLLoss,
             max_epochs=10
             ) -> tuple:

    model.train()
    likelihood.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = loss_cls(model, likelihood)
    # optimizer = torch.optim.LBFGS(model.parameters(), max_iter=1000, lr=0.0001)

    def closure():
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = loss_fn(output, train_y)
        print('Iter {}/{} - Loss: {} noise: {}'.format(
            closure.epoch + 1, 
            max_epochs, 
            loss.item(),
            # model.kernel.l,
            model.likelihood.noise,
            # model.kernel.std,
        ))
        loss.backward(retain_graph=True)
        closure.epoch += 1
        return loss

    closure.epoch = 0
    for epoch in range(max_epochs):
        optimizer.step(closure)

    return model, likelihood

def test_train_gp():
    pass

if __name__ == "__main__":
    test_train_gp()


