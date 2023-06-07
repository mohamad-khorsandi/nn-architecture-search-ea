import os
from enum import Enum

from torchvision.models import ResNet18_Weights, resnet18, ResNet34_Weights, VGG11_Weights, vgg11, resnet34


class FeatureExtractor(Enum):
    RES_NET18 = (resnet18, ResNet18_Weights, 'RES_NET18')
    RES_NET34 = (resnet34, ResNet34_Weights, 'RES_NET34')
    VGG11 = (vgg11, VGG11_Weights, 'VGG11')

    def __init__(self, model, weights, name):
        self.model = model
        self.weights = weights.DEFAULT
        self.file = os.path.join('extracted_features', name + '.npz')