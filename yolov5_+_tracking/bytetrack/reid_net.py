"""
.py created by: cesc
.py created on: 01/05/2020
.py created for: bytetrack => adaptation to work with 5 channels: rgb, d, i
in here we will create our network (DL model)
"""
import os.path

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms
from datasets import AppleCrops

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock(c_out, c_out), ]
    return nn.Sequential(*blocks)


class ReidAppleNet(nn.Module):
    """
    Implementation of a custom NN to extract the features of the images (re-id network)
    """
    def __init__(self, num_classes=1414, reid=False):
        super(ReidAppleNet, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(5, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(64, 64, 2, False)
        # 32 64 32
        self.layer2 = make_layers(64, 128, 2, True)
        # 64 32 16
        self.layer3 = make_layers(128, 256, 2, True)
        # 128 16 8
        self.layer4 = make_layers(256, 1414, 2, True)
        #self.layer4 = make_layers(256, 512, 2, True)
        # 256 8 4
        # todo: aquí se puede mirar de dejar un vector mas grande
        self.avgpool = nn.AvgPool2d((2, 2), 1)
        # 256 1 1
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(1414, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        return x


class ReidAppleNetTriplet(nn.Module):
    """
    Implementation of the Reid AppleNet model with triplet loss from custom NN (ReidAppleNet).
    """
    def __init__(self, num_classes=1414, reid=False):
        super(ReidAppleNetTriplet, self).__init__()
        self.reid_apple_net = ReidAppleNet(num_classes, reid)

    def forward(self, data):
        embedded_x = self.reid_apple_net(data[:, 0, :, :, :]) # batchsize, anchor, channels, height, width
        embedded_y = self.reid_apple_net(data[:, 1, :, :, :]) # batchsize, positive, channels, height, width
        embedded_z = self.reid_apple_net(data[:, 2, :, :, :]) # batchsize, negative, channels, height, width

        # concatenate the three tensors
        embedded = torch.cat((embedded_x, embedded_y, embedded_z), 1)

        return embedded


class ReidAppleNetTripletResNet(nn.Module):
    """
    Implementation of the Reid AppleNet model with triplet loss from resNet pretrained with imagnet but with modified
    input and outputs.
    """
    def __init__(self, num_classes=1414):
        super(ReidAppleNetTripletResNet, self).__init__()
        self.reid_apple_net = load_resnet_modified(num_output_channels=num_classes)

    def forward(self, data):
        embedded_x = self.reid_apple_net(data[:, 0, :, :, :])
        embedded_y = self.reid_apple_net(data[:, 1, :, :, :])
        embedded_z = self.reid_apple_net(data[:, 2, :, :, :])

        # concatenate the three tensors
        embedded = torch.cat((embedded_x, embedded_y, embedded_z), 1)

        return embedded


def load_resnet(model_name='resnet152', pretrained=True):
    """
    Load a pretrained resnet model. The model name should be one of the following:
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    :param model_name: the name of the model
    :param pretrained: if True, load the pretrained weights
    """
    return torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained)


def modify_input_resnet(model, num_input_channels=5):
    """
    Modify the input of the resnet model.
    :param model: the resnet model
    """
    model.conv1 = nn.Conv2d(num_input_channels, 64, 7, 2, 3, bias=False)
    model.bn1 = nn.BatchNorm2d(64)
    model.relu = nn.ReLU(inplace=True)
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


def modify_output_resnet(model, num_output_channels=1414):
    """
    Modify the output of the resnet model.
    :param model: the resnet model
    """
    # model.add_module("fc_last", nn.Linear(1000, num_output_channels))
    model = torch.nn.Sequential(model, torch.nn.Linear(1000, num_output_channels))
    return model


def load_resnet_modified(model_name='resnet152', pretrained=True, num_input_channels=5, num_output_channels=1414):
    """
    Changing a resnet to extract the features of the images (re-id network)
    Load a pretrained resnet model. The model name should be one of the following:
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    :param model_name: the name of the model
    :param pretrained: if True, load the pretrained weights
    """

    net = load_resnet(model_name, pretrained)
    modify_input_resnet(net, num_input_channels)
    net = modify_output_resnet(net, num_output_channels)

    return net


def load_model(model_name):
    """
    Load a trained model. The model name should be one of the following:
    'reid_applenet', 'reid_applenet_triplet', 'reid_applenet_resnet', 'reid_applenet_resnet_triplet'
    :param model_name: the name of the model
    :return: the model loaded
    """
    path_to_model = os.path.join(os.path.dirname(__file__), 'models', f'{model_name}.pth')  # load model from path .pth

    if model_name == 'reid_applenet_resnet':
        # load resnet class
        network = load_resnet_modified()
    elif model_name == 'reid_applenet':
        # load reid class
        network = ReidAppleNet()
    elif model_name == 'reid_applenet_triplet':
        # load reid class
        network = ReidAppleNetTriplet()
    elif model_name == 'reid_applenet_resnet_triplet':
        # load reid class
        network = ReidAppleNetTripletResNet()
    else:
        raise ValueError(f'Model {model_name} not found')

    # load model
    network.load_state_dict(torch.load(path_to_model))

    return network


def infer_batch(network, modelname, data, fmap=True):
    """
    Infer the model on a batch of data.
    :param network: the model
    :param modelname: the name of the model
    :param data: the data
    :return: the output of the model
    """

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        data = data.to('cuda').float()
        network.to('cuda')

    # perform inference of the batch
    with torch.no_grad():
        network.eval()
        # get a feature map if fmap is True
        if fmap:
            if modelname == 'reid_applenet_resnet':
                network = torch.nn.Sequential(*(list(network.children())[:-1])) # delete last fc layer
                network = network[0] # get the resnet model (before was resnet + fc layer => 2 sequential layers)
                network = torch.nn.Sequential(*(list(network.children())[:-2])) # delete fc and avg pool
            elif modelname == 'reid_applenet':
                network = torch.nn.Sequential(*(list(network.children())[:-2])) # delete fc and avg pool
            # todo: no tengo claro como hay que pasarle los datos para que me de el feature map
            elif modelname == 'reid_applenet_triplet' or modelname == 'reid_applenet_resnet_triplet':
                network = network
            else:
                raise ValueError(f'Model {modelname} not found')

    return network(data)


if __name__ == "__main__":
    import time
    model_name = 'reid_applenet_resnet'
    # compute time
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.034, 0.036],   # last 2 values of the vector computed with function
            std=[0.229, 0.224, 0.225, 0.010, 0.008]),   # mean_and_std_calculator in datasets.py
    ])
    db = AppleCrops(root_path='../../data',
                    split='train',
                    transform=transformations)

    img, _ = db.__getitem__(600)

    model = load_model(model_name=model_name)
    input_batch = img.unsqueeze(0)

    output = infer_batch(model, model_name, input_batch)
    print('finished')

