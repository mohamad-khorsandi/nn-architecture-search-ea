import random

from utils import arc_utils
from utils.random_utils import possible_activations, possible_neuron_count, get_rand_neuron_count, get_rand_act, bool_rand


class NetworkArc:
    def __init__(self, feature_extractor):
        self._neurons = list()
        self._activations = list()
        self._feature_extractor = feature_extractor
        self._lock = False
        self._fitness = float()

    def add_layer(self, neuron_count, activation):
        assert not self._lock
        assert activation in possible_activations
        assert neuron_count in possible_neuron_count
        assert len(self._neurons) < 2

        self._neurons.append(neuron_count)
        self._activations.append(activation)
        return self

    def get_fitness(self):
        if self._lock:
            return self._fitness

        self._fitness = arc_utils.evaluate(self)
        self._lock = True
        return self._fitness

    def get_layer_count(self):
        return len(self._neurons)

    def get_feature_extractor(self):
        return self._feature_extractor

    def rand_act(self):
        return random.choice(self._activations)

    def rand_neuron_count(self):
        return random.choice(self._neurons)

    def copy(self):
        cp_arc = NetworkArc(self._feature_extractor)
        for i in range(self.get_layer_count()):
            cp_arc.add_layer(self._neurons[i], self._activations[i])
        return cp_arc

    def get_layer(self, i):
        return self._neurons[i], self._activations[i]

    def mut_layer(self, i, prob):
        if i < self.get_layer_count():
            neuron_count = self._neurons[i]
            act = self._activations[i]

            if bool_rand(prob):
                neuron_count = get_rand_neuron_count()
            if bool_rand(prob):
                act = get_rand_act()
            return neuron_count, act

        else:
            return get_rand_neuron_count(), get_rand_act()