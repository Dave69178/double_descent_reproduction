import torch
from torch.utils.data import DataLoader


def get_number_of_parameters(num_hidden_units):
    """
    Following Belkin's definition for number of parameters in the network for MNIST (dimension = 784, number of classes = 10),
    get the number of parameters in a model with num_hidden_units number of hidden units.
    params:
        num_hidden_units: (int)
    return (int)
    """
    return (784 + 1) * num_hidden_units + (num_hidden_units + 1) * 10


def get_number_of_hidden_units(num_parameters):
    """
    Following Belkin's definition for number of parameters in the network for MNIST (dimension = 784, number of classes = 10),
    get the number of hidden units that would result in a model with num_parameters number of parameters.
    params:
        num_parameters: (int)
    return (float)
    """
    return (num_parameters - 10) / 794


def one_hot_transform(y):
        """
        Transform to convert class labels to one-hot representation
        params:
            y: [list : int]
        return (tensor, n x num_classes)
        """
        return torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)


def count_parameters(model):
    """
    Get count of all tuneable parameters in a pytorch model
    params:
        model: (torch model)
    return: (int)
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_data_to_device(train_dataset, test_dataset, device, train_kwargs, test_kwargs):
    """
    Given a suitably sized dataset, transfer to device before training and return related DataLoader objects.
    As MNIST is small, saves data transfer on every train/test iteration.
    params:
        train_dataset:
        test_dataset:
        device:
        train_kwargs:
        test_kwargs:
    return: (DataLoader, Dataloader)
    """
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    _, (images, labels) = next(enumerate(train_loader))
    images, labels = images.to(device), labels.to(device)
    train_loader = DataLoader(torch.utils.data.TensorDataset(images, labels), **train_kwargs)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    _, (images, labels) = next(enumerate(test_loader))
    images, labels = images.to(device), labels.to(device)
    test_loader = DataLoader(torch.utils.data.TensorDataset(images, labels), **test_kwargs)
    
    return train_loader, test_loader
