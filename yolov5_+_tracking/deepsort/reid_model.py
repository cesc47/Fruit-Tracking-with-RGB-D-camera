import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import logging
import torchvision.transforms as transforms

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


class Net(nn.Module):
    def __init__(self, num_classes=751, reid=False):
        super(Net, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
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
        self.layer4 = make_layers(256, 512, 2, True)
        # 256 8 4
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        # 256 1 1
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
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


class BasicBlock2(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock2, self).__init__()
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


def make_layers2(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock2(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock2(c_out, c_out), ]
    return nn.Sequential(*blocks)


class ReidAppleNet(nn.Module):
    """
    Implementation of a custom NN to extract the features of the images (re-id network)
    """
    def __init__(self, num_classes=914, reid=False):
        super(ReidAppleNet, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(5, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(64, 64, 2, False)
        # 32 64 32
        self.layer2 = make_layers(64, 128, 2, True)
        # 64 32 16
        self.layer3 = make_layers(128, 256, 2, True)
        # 128 16 8
        self.layer4 = make_layers(256, num_classes, 2, True)
        # 256 8 4
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
    def __init__(self, num_classes=914):
        super(ReidAppleNetTripletResNet, self).__init__()
        self.reid_apple_net = load_resnet_modified(num_output_channels=num_classes)

    def forward(self, data):
        embedded_x = self.reid_apple_net(data[:, 0, :, :, :])
        embedded_y = self.reid_apple_net(data[:, 1, :, :, :])
        embedded_z = self.reid_apple_net(data[:, 2, :, :, :])

        # concatenate the three tensors
        embedded = torch.cat((embedded_x, embedded_y, embedded_z), 1)

        return embedded


class ReidAppleNetTripletResNetRGB(nn.Module):
    """
    Implementation of the Reid AppleNet model with triplet loss from resNet pretrained with imagnet but with modified
    input and outputs.
    """
    def __init__(self, num_classes=914):
        super(ReidAppleNetTripletResNetRGB, self).__init__()
        self.reid_apple_net = load_resnet_modified(num_input_channels=3, num_output_channels=num_classes)

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
    :param num_input_channels: the number of input channels
    """
    model.conv1 = nn.Conv2d(num_input_channels, 64, 7, 2, 3, bias=False)
    model.bn1 = nn.BatchNorm2d(64)
    model.relu = nn.ReLU(inplace=True)
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


def modify_output_resnet(model, num_output_channels=914):
    """
    Modify the output of the resnet model.
    :param model: the resnet model
    :param num_output_channels: the number of output channels
    :return: the modified resnet model
    """
    # model.add_module("fc_last", nn.Linear(1000, num_output_channels))
    model = torch.nn.Sequential(model, torch.nn.Linear(1000, num_output_channels))
    return model


def load_resnet_modified(model_name='resnet152', pretrained=True, num_input_channels=5, num_output_channels=914):
    """
    Changing a resnet to extract the features of the images (re-id network)
    Load a pretrained resnet model. The model name should be one of the following:
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    :param model_name: the name of the model
    :param pretrained: if True, load the pretrained weights
    :param num_input_channels: the number of input channels
    :param num_output_channels: the number of output channels
    :return: the modified resnet model
    """

    net = load_resnet(model_name, pretrained)

    if num_input_channels != 3:
        modify_input_resnet(net, num_input_channels)

    net = modify_output_resnet(net, num_output_channels)

    return net


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.model_name = model_path.split('/')[-1].split('.')[0]
        # default network
        if not model_path.endswith('.pth'):
            self.net = Net(reid=True)
            self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
            state_dict = torch.load(model_path, map_location=torch.device(self.device))[
                'net_dict']
            self.net.load_state_dict(state_dict)
            logger = logging.getLogger("root.tracker")
            logger.info("Loading weights from {}... Done!".format(model_path))
            self.net.to(self.device)
            self.size = (64, 128)
            self.norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        else:
            if model_path.endswith('rgb.pth'):
                self.net = load_resnet_modified(num_input_channels=3)
            elif model_path.endswith('resnet.pth'):
                self.net = load_resnet_modified()
            elif model_path.endswith('resnet_triplet.pth'):
                self.net = ReidAppleNetTripletResNet()
            elif model_path.endswith('resnet_triplet_rgb.pth'):
                self.net = ReidAppleNetTripletResNetRGB()
            elif model_path.endswith('resnet_triplet_125.pth'):
                self.net = ReidAppleNetTripletResNet(num_classes=343)
            else:
                raise ValueError("Unknown model type")

            self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
            self.net.load_state_dict(torch.load(model_path), strict=False)
            self.net.to(self.device)
            self.size = (32, 32)
            # reid custom network w/ 5 channels
            if not model_path.endswith('rgb.pth'):
                self.norm = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406, 0.114, 0.073],  # last 2 values of the vector computed with function
                        std=[0.229, 0.224, 0.225, 0.135, 0.0643]),  # mean_and_std_calculator in datasets.py
                ])
            # reid custom network w/ 3 channels
            else:
                self.norm = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],  # last 2 values of the vector computed with function
                        std=[0.229, 0.224, 0.225]),  # mean_and_std_calculator in datasets.py
                ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            if self.model_name.endswith('triplet') or self.model_name.endswith('triplet_125'):
                # add 1 dim to the im_batch to match the input of the network
                im_batch = im_batch.unsqueeze(0)
                # transpose img (a, b, c, d, e) => (b, a, c, d, e)
                im_batch = im_batch.permute(1, 0, 2, 3, 4)
                # replicate the image to match the input of the network
                im_batch = im_batch.repeat(1, 3, 1, 1, 1)
            features = self.net(im_batch)
            if self.model_name.endswith('triplet'):
                # get only the output from the first network (triplet)
                features = features[:, :914]
            elif self.model_name.endswith('125'):
                features = features[:, :343]

        return features.cpu().numpy()