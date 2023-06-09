import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import Linear
from torch.nn.init import xavier_uniform_


class DenseNN(nn.Module):
    """
    Fully-connected neural network with 1 hidden layer for MNIST classification
    """
    def __init__(self, num_hidden_units, activation_fun, device, weights=None, biases=None, glorot_init=False):
        """
        params:
            num_hidden_units: Number of nodes within the hidden layer
            activation_fun: activation function for hidden layer
            device:
            weights: [List : torch.nn.parameter.Parameter] Weights to be used for initialisation of layers. None if random initilisation.
            glorot_init: If no weights supplied, should glorot-uniform dist be used to initialise weights
        """
        super(DenseNN, self).__init__()
        self.l1 = nn.Linear(784, num_hidden_units)
        self.activation1 = activation_fun
        self.l2 = nn.Linear(num_hidden_units, 10)

        if activation_fun is None:
            self.layers = [self.l1, self.l2]
        else:
            self.layers = [self.l1, self.activation1, self.l2]

        if weights is None:
            if glorot_init:
                xavier_uniform_(self.l1.weight)
                xavier_uniform_(self.l2.weight)
        else:
            init_weights(self, weights, biases, device)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    """
    Run single epoch of training on model
    params:
        args:
        model:
        device:
        train_loader:
        optimiser:
        epoch:
    return: (float) training loss
    """
    model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.no_pre_transfer:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target, reduction="sum") / len(data)
        epoch_loss += F.mse_loss(output, target, reduction="sum")
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % args.log_interval == 0:
            logging.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader)}%)]\tLoss: {loss.item()}')
            if args.verbosity > 1:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader)}%)]\tLoss: {loss.item()}')
            if args.dry_run:
                break
    if args.verbosity > 1:
        print(f"epoch loss: {epoch_loss / len(train_loader.dataset)}")
    logging.info(f"epoch loss: {epoch_loss / len(train_loader.dataset)}")
    return epoch_loss.item() / len(train_loader.dataset)


def test(args, model, device, test_loader):
    """
    Test model on test data and get loss
    params:
        model:
        device:
        test_loader:
    return: (float) test loss
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.no_pre_transfer:
                data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(torch.argmax(target, dim=1)).sum().item()   
    test_loss /= len(test_loader.dataset)
    logging.info(f'Test set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)\n')
    if args.verbosity > 1:
        print(f'\nTest set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)\n')
    return test_loss


def init_weights(model, weights, biases, device):
    """
    To be used for the weight reuse scheme. Use weights of previous model (with n nodes) as initialisation of first n nodes of current model (m nodes, m > n).
    Remaining m - n nodes are initialised from a normal(0, 0.01) distribution.
    params:
        model: (pytorch model) Model to initialise weights in
        weights: [list : Tensor] Weights of each layer to set
        device:
    return: (pytorch model)
    """
    hidden_unit_diff = len(model.l1.weight) - len(weights[0])
    ind = 0
    for layer in model.layers:
        if type(layer) == Linear:
            if ind == 0:
                weight_rows = hidden_unit_diff
                weight_cols = 784
                dim = 0
                bias_diff = hidden_unit_diff
            elif ind == 1:
                weight_rows = 10
                weight_cols = hidden_unit_diff
                bias_diff = 0
                dim = 1
            layer.weight = Parameter(torch.cat((weights[ind], torch.normal(0, 0.1, size=(weight_rows, weight_cols)).to(device)), dim=dim))
            layer.bias = Parameter(torch.cat((biases[ind], torch.empty(bias_diff).normal_(0, 0.1).to(device))))
            ind += 1
