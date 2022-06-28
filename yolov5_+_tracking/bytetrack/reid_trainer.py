"""
.py created by: cesc
.py created on: 01/05/2020
.py created for: bytetrack => adaptation to work with 5 channels: rgb, d, i
in here we will train the reid network
"""
import torch
import os
import wandb

from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from os import path
from torch.optim import lr_scheduler

from reid_net import ReidAppleNet
from datasets import AppleCrops
from train import fit


def main():
    wandb.init(project="reid_training", entity="cesc47")

    # cuda management
    DEVICE = 'cuda'
    cuda = torch.cuda.is_available()

    # Find which device is used
    if cuda and DEVICE == "cuda":
        print(f'Training the model in {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        print('CAREFUL!! Training the model with CPU')

    # Output directory
    OUTPUT_MODEL_DIR = './models/'

    # root path of the project
    ROOT_PATH = "../../data"

    # id of the model
    model_id = 'reid_applenet'

    # Create the output directory if it does not exist
    if not path.exists(OUTPUT_MODEL_DIR):
        os.makedirs(OUTPUT_MODEL_DIR)

    # the necessary transformations to work with the dataset
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.5, 0.5],
            std=[0.229, 0.224, 0.225, 0.5, 0.5]),
    ])

    # instantiation of the train and test classes
    train_db = AppleCrops(root_path=ROOT_PATH,
                          split='train',
                          transform=transformations)
    test_db = AppleCrops(root_path=ROOT_PATH,
                         split='test',
                         transform=transformations)

    # creation of the dataloaders
    batch_size = 32
    train_loader = DataLoader(train_db,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)

    test_loader = DataLoader(test_db,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4)

    # instantiation of our network
    model = ReidAppleNet(reid=True)

    # Check if file exists
    if path.exists(OUTPUT_MODEL_DIR + model_id + '.pth'):
        print('Loading the model from the disk')
        model.load_state_dict(torch.load(OUTPUT_MODEL_DIR + model_id + '.pth'))

    # pass the model to the cuda
    if cuda:
        model.cuda()

    # HYPERPARAMS - the hyperparameters of the network: learning rate, number of epochs, optimizer, loss...
    loss_fn = nn.CrossEntropyLoss()
    lr = 3e-4
    log_interval = 10
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    epochs = 5

    # to show the progress of the training in wandb
    wandb.config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size
    }
    # todo: reduce learning rate => DONE
    #  look at normalization of the data (revisar en 'datasets') . if not, gradient clipping?
    # training loop
    fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, epochs, cuda, log_interval, model_id)


if __name__ == "__main__":
    main()