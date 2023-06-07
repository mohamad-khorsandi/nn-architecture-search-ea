import random
from feature_extractor import FeatureExtractor

possible_activations = ['sigmoid', 'relu']

possible_layer_count = [0, 1, 2]

possible_neuron_count = [10, 20, 30]

possible_feature_extractors = [FeatureExtractor.RES_NET18, FeatureExtractor.RES_NET34, FeatureExtractor.VGG11]


def get_rand_extractor():
    return random.choice(possible_feature_extractors)


def get_rand_neuron_count():
    return random.choice(possible_neuron_count)


def get_rand_layer_count():
    return random.choice(possible_layer_count)


def get_rand_act():
    return random.choice(possible_activations)


def bool_rand(probTrue):
    return random.choices([False, True], [1 - probTrue, probTrue])[0]
