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

from reid_net import ReidAppleNet, ReidAppleNetTriplet
from datasets import AppleCrops, AppleCropsTriplet
from train import fit, fit_triplet

from pytorch_metric_learning import distances, losses, miners, reducers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

#https://github.com/pytorch/examples/blob/main/imagenet/main.py => example training imagenet!

def main():

    wandb.init(project="reid_training", entity="cesc47")

    # type of network: reid or reid_triplet
    network = 'reid_triplet'

    # raise error if network is not reid or reid_triplet
    if network not in ['reid', 'reid_triplet']:
        raise ValueError('network must be either reid or reid_triplet')

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
    if network == 'reid':
        model_id = 'reid_applenet'
    else:
        model_id = 'reid_applenet_triplet'

    # Create the output directory if it does not exist
    if not path.exists(OUTPUT_MODEL_DIR):
        os.makedirs(OUTPUT_MODEL_DIR)

    # the necessary transformations to work with the dataset:
    # Often, you want values to have a mean of 0 and a standard deviation of 1 like the standard normal distribution.
    # This is achieved by mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) in rgb images.
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.034, 0.036],   # last 2 values of the vector computed with function
            std=[0.229, 0.224, 0.225, 0.010, 0.008]),   # mean_and_std_calculator in datasets.py
    ])

    # instantiation of the train and test classes
    if network == 'reid':
        train_db = AppleCrops(root_path=ROOT_PATH,
                              split='train',
                              transform=transformations)
        test_db = AppleCrops(root_path=ROOT_PATH,
                             split='test',
                             transform=transformations)
    else:
        train_db = AppleCropsTriplet(root_path=ROOT_PATH,
                                     split='train',
                                     transform=transformations)
        test_db = AppleCropsTriplet(root_path=ROOT_PATH,
                                    split='test',
                                    transform=transformations)

    # creation of the dataloaders
    batch_size = 1024

    train_loader = DataLoader(train_db,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)

    test_loader = DataLoader(test_db,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4)

    # instantiation of our network
    if network == 'reid':
        model = ReidAppleNet()
    else:
        model = ReidAppleNetTriplet()

    # Check if file exists
    if path.exists(OUTPUT_MODEL_DIR + model_id + '.pth'):
        print('Loading the model from the disk')
        model.load_state_dict(torch.load(OUTPUT_MODEL_DIR + model_id + '.pth'))

    # pass the model to the cuda
    if cuda:
        model.cuda()

    # HYPERPARAMS - the hyperparameters of the network: learning rate, number of epochs, optimizer, loss...
    epochs = 30
    lr = 3e-4
    log_interval = 10
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # to show the progress of the training in wandb
    wandb.config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size
    }

    if network == 'reid':
        loss_fn = nn.CrossEntropyLoss()
        scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

        # training loop
        fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, epochs, cuda, log_interval, model_id)

    else:
        # pytorch-metric-learning stuff
        distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low=0)
        loss_fn = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
        mining_func = miners.TripletMarginMiner(
            margin=0.2, distance=distance, type_of_triplets="semihard"
        )
        accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

        fit_triplet(epochs, model, loss_fn, mining_func, DEVICE, train_loader, optimizer, accuracy_calculator,
                    train_db, test_db)


if __name__ == "__main__":
    main()