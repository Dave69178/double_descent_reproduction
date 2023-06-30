import argparse
import os
import logging
import json
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.linear import Linear
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from utils import get_number_of_parameters, get_number_of_hidden_units, one_hot_transform, count_parameters, load_data_to_device
from neural_net import DenseNN, train, test


def main():
    parser = argparse.ArgumentParser(description='Belkin double descent reproduction')
    parser.add_argument('--hidden-units', nargs="*", type=int, default=52, metavar='N',
                        help='number of nodes within hidden layer (default: 4)')
    parser.add_argument('--activation-fun', type=str, default="sigmoid",
                        help='activation function to be used (default: sigmoid) (options: sigmoid, relu, none)')
    parser.add_argument('--train-size', type=int, default=4000,
                        help='size of training dataset (default: 4000)')
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
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.9)')
    parser.add_argument('--weight-reuse', action='store_true', default=False,
                        help='Initialise larger models with smaller model final weights (default: False)')
    parser.add_argument('--glorot-init', action='store_true', default=False,
                        help='Initialise first model weights with glorot-uniform dist (default: False)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--no-pre-transfer', action='store_true', default=False,
                        help='disable transfer of data to device before train/test loops (default: False)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass (default: False)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status (default: 20)')
    parser.add_argument('--verbosity', type=int, default=1,
                        help='Amount of detail to be printed (doesn\'t affect logs) (default: 1) (options: 0 = None, 1 = Indicated completion of training each model, 2 = Show loss throughout training)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model (default: False)')
    parser.add_argument('--save-metrics', action='store_true', default=False,
                        help='For Saving the train and test losses (default: False)')
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

    if args.verbosity > 0:
        print(f"Device: {device}")

    if type(args.hidden_units) == int:
        args.hidden_units = [args.hidden_units]

    if args.weight_reuse:
        last_model_weight = None
        last_model_biases = None

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

    logging.info(f"TRAIN_SET_SIZE: {args.train_size}")
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                    transform=transform, target_transform=transforms.Compose([one_hot_transform]))
    dataset2 = datasets.MNIST('../data', train=False, download=True,
                    transform=transform, target_transform=transforms.Compose([one_hot_transform]))
    dataset1 = Subset(dataset1, np.random.choice(len(dataset1), args.train_size, replace=False))

    if args.no_pre_transfer:
        # For use if not transferring to GPU before training loop
        train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    else:
        # Transfer whole datasets to device (i.e. GPU) before training (Faster)
        train_loader, test_loader = load_data_to_device(dataset1, dataset2, device, train_kwargs, test_kwargs)

    for num in args.hidden_units:
        if args.verbosity > 0:
            print(f"Training model with {num} hidden units")

        # Get date and time for logging
        date_now = str(datetime.now().date())
        time_now = str(datetime.now().time())[:-7].replace(":", ";")

        if args.weight_reuse:
            path = f"./models/{date_now}/weight_reuse/{time_now}/"
        else:
            path = f"./models/{date_now}/no_weight_reuse/{time_now}/"
        os.makedirs(path, exist_ok = True) 

        logging.basicConfig(filename=os.path.join(path, f"train_{time_now}.log"),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG,
                            force=True)
        logging.info(f"{args}")

        if args.activation_fun == "sigmoid":
            activation = nn.Sigmoid()
        elif args.activation_fun == "relu":
            activation = nn.ReLU()
        else:
            activation = None

        NUM_PARAMS = get_number_of_parameters(num)

        if args.weight_reuse and NUM_PARAMS < args.train_size * 10:
            print("using weights")
            model = DenseNN(num, activation, device, last_model_weight, last_model_biases, args.glorot_init).to(device)
        else:
            print("not using weights")
            model = DenseNN(num, activation, device, None, None, False).to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        # Apply learning rate decay to models under interpolation threshold
        if NUM_PARAMS < args.train_size * 10:
            scheduler = StepLR(optimizer, step_size=500, gamma=args.gamma)
        else:
            scheduler = StepLR(optimizer, step_size=500, gamma=1)

        metrics = {
            "train_loss": [],
            "test_loss": []
        }

        for epoch in range(1, args.epochs + 1):
            metrics["train_loss"].append(train(args, model, device, train_loader, optimizer, epoch))
            metrics["test_loss"].append(test(args, model, device, test_loader))

            scheduler.step()

            if metrics["train_loss"][-1] < 0.0001 and NUM_PARAMS < args.train_size * 10:
                break

            if epoch % 500 == 0:
                if args.verbosity > 0:
                    print(f"LEARNING RATE: {scheduler.get_lr()}")
                    logging.info(f"LEARNING RATE: {scheduler.get_lr()}")
                if args.save_model:
                    os.makedirs(path, exist_ok = True)
                    torch.save(model.state_dict(), os.path.join(path, f"model_w_{num}_hidden_units_epoch_{epoch}.pt"))

        if args.save_model:
            torch.save(model.state_dict(), os.path.join(path, f"model_w_{num}_hidden_units_final.pt"))

        if args.save_metrics:
            if args.verbosity > 1:
                print("saving metrics")
            with open(os.path.join(path, f"model_w_{num}_hidden_units_loss_metrics.json"), "w") as outfile:
                outfile.write(json.dumps(metrics, indent=4))

        if args.weight_reuse:
            last_model_weight = []
            last_model_biases = []
            for layer in model.layers:
                if type(layer) == Linear:
                    last_model_weight.append(layer.weight)
                    last_model_biases.append(layer.bias)


if __name__ == '__main__':
    main()