from email import utils

import tensorflow as tf
from tensorflow import keras
from utils.load_and_extract_utils import get_train_features, get_test_features
from network_arc import NetworkArc
from utils.random_utils import get_rand_extractor, get_rand_layer_count, get_rand_neuron_count, get_rand_act

_OUT_PUT_CLASSES = 10
_EPOCHS = 5


def evaluate(arc):
    print('evaluating ->', signature_not_fitness(arc))
    x_train, y_train = get_train_features(arc.get_feature_extractor())
    model = arc2model(arc, x_train.shape[1], _OUT_PUT_CLASSES)
    model.fit(x_train, y_train, epochs=_EPOCHS, verbose=0)
    x_test, y_test = get_test_features(arc.get_feature_extractor())
    loss, acu = model.evaluate(x_test, y_test, verbose=0)

    print('accuracy: ', round(acu, 4))
    print()
    return acu


def arc2model(arc, input_layers, output_layers):
    model = tf.keras.Sequential()
    model.add(keras.Input(shape=(input_layers,)))
    for i in range(arc.get_layer_count()):
        neurons, act = arc.get_layer(i)
        model.add(keras.layers.Dense(neurons, activation=act))
    model.add(keras.layers.Dense(output_layers, activation='linear'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def signature(arc: NetworkArc):
    res = str(arc.get_feature_extractor().file)[-9:-4]
    for i in range(arc.get_layer_count()):
        layer = arc.get_layer(i)
        res += ':' + str(layer[0]) + layer[1][0]
    res += '(' + str(round(arc.get_fitness(), 2)) + ')'
    return res


def signature_not_fitness(arc: NetworkArc):
    res = str(arc.get_feature_extractor().file)[-9:-4]
    for i in range(arc.get_layer_count()):
        layer = arc.get_layer(i)
        res += ':' + str(layer[0]) + layer[1][0]
    res += '()'
    return res

def get_random_arc():
    feature_extractor = get_rand_extractor()
    rad_arc = NetworkArc(feature_extractor)

    layer_count = get_rand_layer_count()
    for i in range(layer_count):
        neurons = get_rand_neuron_count()
        act = get_rand_act()
        rad_arc.add_layer(neurons, act)

    return rad_arc
