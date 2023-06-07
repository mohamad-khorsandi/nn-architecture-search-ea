import os
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms


def get_train_features(fx):
    if os.path.exists(fx.file):
        data = np.load(fx.file)
        x_train = data['x_train']
        y_train = data['y_train']
        return x_train, y_train
    else:
        x_train, y_train, x_test, y_test = fx._extract_and_save()
        return x_train, y_train


def get_test_features(fx):
    if os.path.exists(fx.file):
        data = np.load(fx.file)
        x_test = data['x_test']
        y_test = data['y_test']
        return x_test, y_test

    else:
        x_train, y_train, x_test, y_test = fx._extract_and_save()
        return x_test, y_test


def _extract_and_save(fx):
    train_images, y_train, test_images, y_test = load_cifar10()
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    x_train = extract_feature(fx, train_images)
    x_test = extract_feature(fx, test_images)
    np.savez(fx.file, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return x_train, y_train, x_test, y_test


def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = datasets.CIFAR10('data', train=True,
                                 download=True, transform=transform)
    test_set = datasets.CIFAR10('data', train=False,
                                download=True, transform=transform)

    train_images, y_train = get_images_and_labels(train_set)
    test_images, y_test = get_images_and_labels(test_set)

    return train_images, y_train, test_images, y_test


def get_images_and_labels(data_set):
    images = []
    labels = []
    for image, label in data_set:
        images.append(image)
        labels.append(label)

    return images, labels


def extract_feature(fx, images):
    images = torch.stack(images)
    model = fx.model(weights=fx.weights)

    for param in model.parameters():
        param.requires_grad = False
    modules = list(model.children())[:-1]
    model = torch.nn.Sequential(*modules)
    features = model(images).numpy()
    return features.reshape(features.shape[0], -1)
