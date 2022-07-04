"""
.py created by: cesc
.py created on: 01/05/2020
.py created for: bytetrack => adaptation to work with 5 channels: rgb, d, i
in here we will create our network (DL model)
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms

"""
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
    def __init__(self, num_classes=751, reid=False):
        super(ReidAppleNet, self).__init__()
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
"""


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
    def __init__(self, num_classes=1414, reid=False):
        super(ReidAppleNetTriplet, self).__init__()
        self.reid_apple_net = ReidAppleNet(num_classes, reid)

    def forward(self, x, y, z):
        embedded_x = self.reid_apple_net(x)
        embedded_y = self.reid_apple_net(y)
        embedded_z = self.reid_apple_net(z)

        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)

        return dist_a, dist_b, embedded_x, embedded_y, embedded_z


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
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.fc = nn.Linear(512, num_output_channels)


def load_resnet_modified(model_name='resnet152', pretrained=True, num_input_channels=5, num_output_channels=1414):
    """
    Load a pretrained resnet model. The model name should be one of the following:
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    :param model_name: the name of the model
    :param pretrained: if True, load the pretrained weights
    """

    net = load_resnet(model_name, pretrained)
    modify_input_resnet(net, num_input_channels)
    modify_output_resnet(net, num_output_channels)

    return net


if __name__ == "__main__":
    """
    net = ReidAppleNet(reid=True)
    print(net)
    print('---')
    net_triplet = ReidAppleNetTriplet(reid=True)
    print(net_triplet)
    print('---')
    """
    model = load_resnet_modified()

    from datasets import AppleCrops
    db = AppleCrops(root_path='../../data',
                    split='train',
                    transform=None)
    img, label = db.__getitem__(600)
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.146, 0.096],   # last 2 values of the vector computed with function
            std=[0.229, 0.224, 0.225, 0.151, 0.102]),   # mean_and_std_calculator in datasets.py
    ])
    input_tensor = transformations(img)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda').float()
        model.to('cuda')

    with torch.no_grad():
        model.eval()
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)

    print('finished')
