import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import tensorflow as tf


# Training param
input_vector_size = 8    # input vactor size, the num of neurons in the input layer
num_class = 2            # how many classes

num_of_nodes_per_layer = 4

keep_prob = 1

################################################################
#--------------- III Define network structure ------------------
neuron_size_layer_1 = 12
neuron_size_layer_2 = 6
neuron_size_layer_3 = 4
neuron_size_layer_4 = 4
neuron_size_layer_5 = num_of_nodes_per_layer
neuron_size_layer_6 = num_of_nodes_per_layer
neuron_size_layer_7 = num_of_nodes_per_layer
neuron_size_layer_8 = num_of_nodes_per_layer
neuron_size_layer_9 = num_of_nodes_per_layer
neuron_size_layer_10 = num_of_nodes_per_layer
neuron_size_layer_11 = num_of_nodes_per_layer
neuron_size_layer_12 = num_of_nodes_per_layer
neuron_size_layer_13 = num_of_nodes_per_layer
neuron_size_layer_14 = num_of_nodes_per_layer
neuron_size_layer_15 = num_of_nodes_per_layer
neuron_size_layer_16 = num_of_nodes_per_layer
neuron_size_layer_17 = num_of_nodes_per_layer
neuron_size_layer_18 = num_of_nodes_per_layer
neuron_size_layer_19 = num_of_nodes_per_layer
neuron_size_layer_20 = num_of_nodes_per_layer
neuron_size_layer_21 = num_of_nodes_per_layer
neuron_size_layer_22 = num_of_nodes_per_layer
neuron_size_layer_23 = num_of_nodes_per_layer
neuron_size_layer_24 = num_of_nodes_per_layer
neuron_size_layer_25 = num_of_nodes_per_layer
neuron_size_layer_26 = num_of_nodes_per_layer
neuron_size_layer_27 = num_of_nodes_per_layer
neuron_size_layer_28 = num_of_nodes_per_layer
neuron_size_layer_29 = num_of_nodes_per_layer
neuron_size_layer_30 = num_of_nodes_per_layer

weights = {
    'h1': tf.Variable(tf.truncated_normal([input_vector_size, neuron_size_layer_1])),  # 1st num of in, 2nd num of out
    'h2': tf.Variable(tf.truncated_normal([neuron_size_layer_1, neuron_size_layer_2])),
    'h3': tf.Variable(tf.truncated_normal([neuron_size_layer_2, neuron_size_layer_3])),
    'h4': tf.Variable(tf.truncated_normal([neuron_size_layer_3, neuron_size_layer_4])),
    'h5': tf.Variable(tf.truncated_normal([neuron_size_layer_4, neuron_size_layer_5])),
    'h6': tf.Variable(tf.truncated_normal([neuron_size_layer_5, neuron_size_layer_6])),
    'h7': tf.Variable(tf.truncated_normal([neuron_size_layer_6, neuron_size_layer_7])),
    'h8': tf.Variable(tf.truncated_normal([neuron_size_layer_7, neuron_size_layer_8])),
    'h9': tf.Variable(tf.truncated_normal([neuron_size_layer_8, neuron_size_layer_9])),
    'h10': tf.Variable(tf.truncated_normal([neuron_size_layer_9, neuron_size_layer_10])),
    'h11': tf.Variable(tf.truncated_normal([neuron_size_layer_10, neuron_size_layer_11])),  # 1st num of in, 2nd num of out
    'h12': tf.Variable(tf.truncated_normal([neuron_size_layer_11, neuron_size_layer_12])),
    'h13': tf.Variable(tf.truncated_normal([neuron_size_layer_12, neuron_size_layer_13])),
    'h14': tf.Variable(tf.truncated_normal([neuron_size_layer_13, neuron_size_layer_14])),
    'h15': tf.Variable(tf.truncated_normal([neuron_size_layer_14, neuron_size_layer_15])),
    'h16': tf.Variable(tf.truncated_normal([neuron_size_layer_15, neuron_size_layer_16])),
    'h17': tf.Variable(tf.truncated_normal([neuron_size_layer_16, neuron_size_layer_17])),
    'h18': tf.Variable(tf.truncated_normal([neuron_size_layer_17, neuron_size_layer_18])),
    'h19': tf.Variable(tf.truncated_normal([neuron_size_layer_18, neuron_size_layer_19])),
    'h20': tf.Variable(tf.truncated_normal([neuron_size_layer_19, neuron_size_layer_20])),
    'h21': tf.Variable(tf.truncated_normal([neuron_size_layer_20, neuron_size_layer_21])),
    'h22': tf.Variable(tf.truncated_normal([neuron_size_layer_21, neuron_size_layer_22])),
    'h23': tf.Variable(tf.truncated_normal([neuron_size_layer_22, neuron_size_layer_23])),
    'h24': tf.Variable(tf.truncated_normal([neuron_size_layer_23, neuron_size_layer_24])),
    'h25': tf.Variable(tf.truncated_normal([neuron_size_layer_24, neuron_size_layer_25])),
    'h26': tf.Variable(tf.truncated_normal([neuron_size_layer_25, neuron_size_layer_26])),
    'h27': tf.Variable(tf.truncated_normal([neuron_size_layer_26, neuron_size_layer_27])),
    'h28': tf.Variable(tf.truncated_normal([neuron_size_layer_27, neuron_size_layer_28])),
    'h29': tf.Variable(tf.truncated_normal([neuron_size_layer_28, neuron_size_layer_29])),
    'h30': tf.Variable(tf.truncated_normal([neuron_size_layer_29, neuron_size_layer_30])),

    'out': tf.Variable(tf.truncated_normal([neuron_size_layer_2, num_class]))
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([neuron_size_layer_1])),
    'b2': tf.Variable(tf.truncated_normal([neuron_size_layer_2])),
    'b3': tf.Variable(tf.truncated_normal([neuron_size_layer_3])),
    'b4': tf.Variable(tf.truncated_normal([neuron_size_layer_4])),
    'b5': tf.Variable(tf.truncated_normal([neuron_size_layer_5])),
    'b6': tf.Variable(tf.truncated_normal([neuron_size_layer_6])),
    'b7': tf.Variable(tf.truncated_normal([neuron_size_layer_7])),
    'b8': tf.Variable(tf.truncated_normal([neuron_size_layer_8])),
    'b9': tf.Variable(tf.truncated_normal([neuron_size_layer_9])),
    'b10': tf.Variable(tf.truncated_normal([neuron_size_layer_10])),

    'b11': tf.Variable(tf.truncated_normal([neuron_size_layer_11])),
    'b12': tf.Variable(tf.truncated_normal([neuron_size_layer_12])),
    'b13': tf.Variable(tf.truncated_normal([neuron_size_layer_13])),
    'b14': tf.Variable(tf.truncated_normal([neuron_size_layer_14])),
    'b15': tf.Variable(tf.truncated_normal([neuron_size_layer_15])),
    'b16': tf.Variable(tf.truncated_normal([neuron_size_layer_16])),
    'b17': tf.Variable(tf.truncated_normal([neuron_size_layer_17])),
    'b18': tf.Variable(tf.truncated_normal([neuron_size_layer_18])),
    'b19': tf.Variable(tf.truncated_normal([neuron_size_layer_19])),
    'b20': tf.Variable(tf.truncated_normal([neuron_size_layer_20])),

    'b21': tf.Variable(tf.truncated_normal([neuron_size_layer_21])),
    'b22': tf.Variable(tf.truncated_normal([neuron_size_layer_22])),
    'b23': tf.Variable(tf.truncated_normal([neuron_size_layer_23])),
    'b24': tf.Variable(tf.truncated_normal([neuron_size_layer_24])),
    'b25': tf.Variable(tf.truncated_normal([neuron_size_layer_25])),
    'b26': tf.Variable(tf.truncated_normal([neuron_size_layer_26])),
    'b27': tf.Variable(tf.truncated_normal([neuron_size_layer_27])),
    'b28': tf.Variable(tf.truncated_normal([neuron_size_layer_28])),
    'b29': tf.Variable(tf.truncated_normal([neuron_size_layer_29])),
    'b30': tf.Variable(tf.truncated_normal([neuron_size_layer_30])),

    'out': tf.Variable(tf.truncated_normal([num_class]))
}

activity_functions = {
    'fun1' : tf.nn.tanh,
    'fun2' : tf.nn.tanh,
    'fun3' : tf.nn.tanh,
    'fun4' : tf.nn.tanh,
    'fun5' : tf.nn.relu,
    'fun6' : tf.nn.relu,
    'fun7' : tf.nn.relu,
    'fun8' : tf.nn.relu,
    'fun9' : tf.nn.relu,
    'fun10' : tf.nn.relu,

    'fun11' : tf.nn.relu,
    'fun12' : tf.nn.relu,
    'fun13' : tf.nn.relu,
    'fun14' : tf.nn.relu,
    'fun15' : tf.nn.relu,
    'fun16' : tf.nn.relu,
    'fun17' : tf.nn.relu,
    'fun18' : tf.nn.relu,
    'fun19' : tf.nn.relu,
    'fun20' : tf.nn.relu,

    'fun21': tf.nn.relu,
    'fun22': tf.nn.relu,
    'fun23': tf.nn.relu,
    'fun24': tf.nn.relu,
    'fun25': tf.nn.relu,
    'fun26': tf.nn.relu,
    'fun27': tf.nn.relu,
    'fun28': tf.nn.relu,
    'fun29': tf.nn.relu,
    'fun30': tf.nn.relu,

    'out' : tf.nn.softmax
}

#--------------- Muti-layer perceptron ------------------
def multilayer_perceptron(x, weights = weights, biases = biases, activity_functions = activity_functions):

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = activity_functions['fun1'](layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = activity_functions['fun2'](layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    #
    # layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # layer_3 = activity_functions['fun3'](layer_3)
    # layer_3 = tf.nn.dropout(layer_3, keep_prob)

    # layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    # layer_4 = activity_functions['fun4'](layer_4)
    # layer_4 = tf.nn.dropout(layer_4, keep_prob)

    # layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    # layer_5 = activity_functions['fun5'](layer_5)
    # layer_5 = tf.nn.dropout(layer_5, keep_prob)
    #
    # layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
    # layer_6 = activity_functions['fun6'](layer_6)
    # layer_6 = tf.nn.dropout(layer_6, keep_prob)
    #
    # layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
    # layer_7 = activity_functions['fun7'](layer_7)
    # layer_7 = tf.nn.dropout(layer_7, keep_prob)
    #
    # layer_8 = tf.add(tf.matmul(layer_7, weights['h8']), biases['b8'])
    # layer_8 = activity_functions['fun8'](layer_8)
    # layer_8 = tf.nn.dropout(layer_8, keep_prob)
    #
    # layer_9 = tf.add(tf.matmul(layer_8, weights['h9']), biases['b9'])
    # layer_9 = activity_functions['fun9'](layer_9)
    # layer_9 = tf.nn.dropout(layer_9, keep_prob)
    #
    # layer_10 = tf.add(tf.matmul(layer_9, weights['h10']), biases['b10'])
    # layer_10 = activity_functions['fun10'](layer_10)
    # layer_10 = tf.nn.dropout(layer_10, keep_prob)
    #
    # layer_11 = tf.add(tf.matmul(layer_10, weights['h11']), biases['b11'])
    # layer_11 = activity_functions['fun11'](layer_11)
    # layer_11 = tf.nn.dropout(layer_11, keep_prob)
    #
    # layer_12 = tf.add(tf.matmul(layer_11, weights['h12']), biases['b12'])
    # layer_12 = activity_functions['fun12'](layer_12)
    # layer_12 = tf.nn.dropout(layer_12, keep_prob)

    # layer_13 = tf.add(tf.matmul(layer_12, weights['h13']), biases['b13'])
    # layer_13 = activity_functions['fun13'](layer_13)
    # layer_13 = tf.nn.dropout(layer_13, keep_prob)
    #
    # layer_14 = tf.add(tf.matmul(layer_13, weights['h14']), biases['b14'])
    # layer_14 = activity_functions['fun14'](layer_14)
    # layer_14 = tf.nn.dropout(layer_14, keep_prob)
    #
    # layer_15 = tf.add(tf.matmul(layer_14, weights['h15']), biases['b15'])
    # layer_15 = activity_functions['fun15'](layer_15)
    # layer_15 = tf.nn.dropout(layer_15, keep_prob)
    #
    # layer_16 = tf.add(tf.matmul(layer_15, weights['h16']), biases['b16'])
    # layer_16 = activity_functions['fun16'](layer_16)
    # layer_16 = tf.nn.dropout(layer_16, keep_prob)
    #
    # layer_17 = tf.add(tf.matmul(layer_16, weights['h17']), biases['b17'])
    # layer_17 = activity_functions['fun17'](layer_17)
    # layer_17 = tf.nn.dropout(layer_17, keep_prob)
    #
    # layer_18 = tf.add(tf.matmul(layer_17, weights['h18']), biases['b18'])
    # layer_18 = activity_functions['fun18'](layer_18)
    # layer_18 = tf.nn.dropout(layer_18, keep_prob)
    #
    # layer_19 = tf.add(tf.matmul(layer_18, weights['h19']), biases['b19'])
    # layer_19 = activity_functions['fun19'](layer_19)
    # layer_19 = tf.nn.dropout(layer_19, keep_prob)
    #
    # layer_20 = tf.add(tf.matmul(layer_19, weights['h20']), biases['b20'])
    # layer_20 = activity_functions['fun20'](layer_20)
    # layer_20 = tf.nn.dropout(layer_20, keep_prob)
    #
    # layer_21 = activity_functions['fun21'](tf.add(tf.matmul(layer_20, weights['h21']), biases['b21']))
    # layer_21 = tf.nn.dropout(layer_21, keep_prob)
    #
    # layer_22 = activity_functions['fun22'](tf.add(tf.matmul(layer_21, weights['h22']), biases['b22']))
    # layer_22 = tf.nn.dropout(layer_22, keep_prob)
    #
    # layer_23 = activity_functions['fun23'](tf.add(tf.matmul(layer_22, weights['h23']), biases['b23']))
    # layer_23 = tf.nn.dropout(layer_23, keep_prob)
    #
    # layer_24 = activity_functions['fun24'](tf.add(tf.matmul(layer_23, weights['h24']), biases['b24']))
    # layer_24 = tf.nn.dropout(layer_24, keep_prob)
    #
    # layer_25 = activity_functions['fun25'](tf.add(tf.matmul(layer_24, weights['h25']), biases['b25']))
    # layer_25 = tf.nn.dropout(layer_25, keep_prob)
    #
    # layer_26 = activity_functions['fun26'](tf.add(tf.matmul(layer_25, weights['h26']), biases['b26']))
    # layer_26 = tf.nn.dropout(layer_26, keep_prob)
    #
    # layer_27 = activity_functions['fun27'](tf.add(tf.matmul(layer_26, weights['h27']), biases['b27']))
    # layer_27 = tf.nn.dropout(layer_27, keep_prob)
    #
    # layer_28 = activity_functions['fun28'](tf.add(tf.matmul(layer_27, weights['h28']), biases['b28']))
    # layer_28 = tf.nn.dropout(layer_28, keep_prob)
    #
    # layer_29 = activity_functions['fun29'](tf.add(tf.matmul(layer_28, weights['h29']), biases['b29']))
    # layer_29 = tf.nn.dropout(layer_29, keep_prob)
    #
    # layer_30 = activity_functions['fun30'](tf.add(tf.matmul(layer_29, weights['h30']), biases['b30']))
    # layer_30 = tf.nn.dropout(layer_30, keep_prob)

    # output layer
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    out_layer = activity_functions['out'](out_layer)
    return out_layer