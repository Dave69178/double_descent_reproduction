import argparse
import os
import logging
import json
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


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
        return torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)

def load_data_to_GPU(train_dataset, test_dataset, device, train_kwargs, test_kwargs):
    # put all data to GPU
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    _, (images, labels) = next(enumerate(train_loader))
    images, labels = images.to(device), labels.to(device)
    train_loader = DataLoader(torch.utils.data.TensorDataset(images, labels), **train_kwargs)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    _, (images, labels) = next(enumerate(test_loader))
    images, labels = images.to(device), labels.to(device)
    test_loader = DataLoader(torch.utils.data.TensorDataset(images, labels), **test_kwargs)
    
    return train_loader, test_loader

class DenseNN(nn.Module):
    def __init__(self, num_hidden_units):
        super(DenseNN, self).__init__()
        self.l1 = nn.Linear(784, num_hidden_units)
        self.sigmoid1 = nn.Sigmoid()
        self.l2 = nn.Linear(num_hidden_units, 10)
        
    def forward(self, x):
        return self.l2(self.sigmoid1(self.l1(x)))


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target, reduction="mean")
        epoch_loss += F.mse_loss(output, target, reduction="sum")
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logging.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader)}%)]\tLoss: {loss.item()}')
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader)}%)]\tLoss: {loss.item()}')
            if args.dry_run:
                break
    print(f"epoch loss: {epoch_loss / len(train_loader.dataset)}")
    return epoch_loss.item() / len(train_loader.dataset)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            #data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1)  # get the index of the max log-probability
            correct += pred.eq(torch.argmax(target, dim=1)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    logging.info(f'Test set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)\n')
    print(f'\nTest set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)\n')
    return test_loss

def main(hidden_units):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--hidden-units', type=int, default=hidden_units, metavar='N',
                        help='number of nodes within hidden layer (default: 4)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=6000, metavar='N',
                        help='number of epochs to train (default: 6000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.95, metavar='LR',
                        help='sgd momentum parameter (default: 0.95)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--save-metrics', action='store_true', default=True,
                        help='For Saving the train and test losses')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Get date and time for logging
    date_now = str(datetime.now().date())
    time_now = str(datetime.now().time())[:-7].replace(":", ";")

    path = f"./models/{date_now}/{time_now}/"
    os.makedirs(path, exist_ok = True) 

    logging.basicConfig(filename=os.path.join(path, f"train_{time_now}.log"),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    
    logging.info(f"{args}")

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0, # Set to 0 as error when using data pre-transferred to GPU
                       'pin_memory': False} # Set to False for same reason as above
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        torch.flatten
        ])

    TRAIN_SET_SIZE = 4000
    logging.info(f"TRAIN_SET_SIZE: {TRAIN_SET_SIZE}")
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform, target_transform=transforms.Compose([one_hot_transform]))
    dataset2 = datasets.MNIST('../data', train=False, download=True,
                       transform=transform, target_transform=transforms.Compose([one_hot_transform]))
    dataset1 = Subset(dataset1, np.random.choice(len(dataset1), TRAIN_SET_SIZE, replace=False))

    # For use if not transferring to GPU before training loop (Must also uncomment line in train and test loop to transfer to GPU)
    #train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    #test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Transfer whole datasets to GPU before training (Faster)
    train_loader, test_loader = load_data_to_GPU(dataset1, dataset2, device, train_kwargs, test_kwargs)

    model = DenseNN(args.hidden_units).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Apply learning rate decay to models under interpolation threshold
    if get_number_of_parameters(args.hidden_units) < TRAIN_SET_SIZE:
        scheduler = StepLR(optimizer, step_size=500, gamma=args.gamma)
    else:
        scheduler = StepLR(optimizer, step_size=500, gamma=1)

    metrics = {
        "train_loss": [],
        "test_loss": []
    }

    for epoch in range(1, args.epochs + 1):
        metrics["train_loss"].append(train(args, model, device, train_loader, optimizer, epoch))
        metrics["test_loss"].append(test(model, device, test_loader))
        scheduler.step()
        if epoch % 500 == 0:
            if args.save_model:
                path = f"./models/{date_now}/{time_now}/"
                os.makedirs(path, exist_ok = True) 
                torch.save(model.state_dict(), os.path.join(path, f"model_w_{args.hidden_units}_hidden_units_epoch_{epoch}.pt"))

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(path, f"model_w_{args.hidden_units}_hidden_units_final.pt"))

    if args.save_metrics:
        print("saving metrics")
        with open(os.path.join(path, f"model_w_{args.hidden_units}_hidden_units_loss_metrics.json"), "w") as outfile:
            outfile.write(json.dumps(metrics, indent=4))


if __name__ == '__main__':
    for num_hidden_units in [4, 10, 19, 31, 44, 53, 63, 113, 379]:#, [4, 6, 10, 13, 19, 25, 31, 38, 44, 50, 53, 57, 63, 76, 113, 252, 378]:
        main(num_hidden_units)