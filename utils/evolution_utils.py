import random
from random import choices

from utils.random_utils import get_rand_layer_count
from network_arc import NetworkArc
from utils.random_utils import bool_rand, get_rand_extractor


def parent_selection(generation, count):
    prob_list = _get_weight_list(generation)
    return choices(generation, prob_list, k=count)


def mutation(parent: NetworkArc, p_mut, mut_step):
    if not bool_rand(p_mut):
        return parent.copy()

    feature_extractor = parent.get_feature_extractor()

    if bool_rand(mut_step):
        feature_extractor = get_rand_extractor()

    result = NetworkArc(feature_extractor)

    layer_count = parent.get_layer_count()
    if bool_rand(mut_step):
        layer_count = get_rand_layer_count()

    for i in range(layer_count):
        neurons, act = parent.mut_layer(i, mut_step)
        result.add_layer(neurons, act)

    return result


def recombination(p1: NetworkArc, p2: NetworkArc, p_rec):
    if not bool_rand(p_rec):
        return p1.copy(), p2.copy()

    children = []
    for j in range(2):
        layer_count = random.choice([p1.get_layer_count(), p2.get_layer_count()])
        feature_extractor = random.choice([p1.get_feature_extractor(), p2.get_feature_extractor()])
        child = NetworkArc(feature_extractor)
        for _ in range(layer_count):
            rand_p = get_rand_parent_with_layer(p1, p2)
            i_th_layer_neurons = rand_p.rand_neuron_count()

            rand_p = get_rand_parent_with_layer(p1, p2)
            i_th_layer_act = rand_p.rand_act()

            child.add_layer(i_th_layer_neurons, i_th_layer_act)
        children.append(child)

    return children


def get_rand_parent_with_layer(p1:NetworkArc, p2:NetworkArc):
    if p1.get_layer_count() == 0:
        return p2
    elif p2.get_layer_count() == 0:
        return p1
    return random.choice([p1, p2])


def _get_weight_list(chromosome_list: list, reverse=False):
    fittness_list = [c.get_fitness() for c in chromosome_list]

    fittness_sum = sum(fittness_list)
    assert fittness_sum != 0

    weight_list = [p / fittness_sum for p in fittness_list]

    assert any([0 <= p <= 1 for p in weight_list])
    if reverse:
        return [1 - p for p in weight_list]
    else:
        return weight_list
