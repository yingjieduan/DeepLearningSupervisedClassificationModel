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

from utility.io_operation import read_dataset, one_hot_encode
from perceptron_network_design import multilayer_perceptron


################################################################
#-------------- I System configuration -------------------------
# Input and Output folders
input_data = "./resource/diabetes.csv"
model_path = "./output/DeepLearningModel/model"
output_standardization_scaler = "./output/Standardization/scaler.pkl"

output_folder = "./output/"

output = "./output"
feature_index = 8
label_index = 8

input_vector_size = 8    # input vactor size, the num of neurons in the input layer
num_class = 2            # how many classes

X, Y, original_label= read_dataset(input_data, feature_index, label_index)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# training system input placehoder
x = tf.placeholder(tf.float32, [None, input_vector_size])  # placeholder is input: None - how many vectors; n_dim - dimensions of vectors
y_ = tf.placeholder(tf.float32, [None, num_class])  # label placehoder
y = multilayer_perceptron(x)

sess = tf.Session()
sess.run(init)
saver.restore(sess, model_path)
scaler = None
if os.path.isfile(output_standardization_scaler):
    scaler = joblib.load(output_standardization_scaler)

prediction = tf.argmax(y, 1)
correct_predict_result = []

print('------------------------------ Prediction ---------------------------')
for i in range(len(X)):
    input_vector = np.array([X[i]])
    input_vector = scaler.transform(input_vector) if scaler is not None else input_vector

    prediction_run = sess.run(prediction, feed_dict = {x: input_vector.reshape(1,input_vector_size)})
    print('Original Class:', original_label[i], '\t predicted:', prediction_run[0],
          '\t correct_detect:', original_label[i]==prediction_run[0])
    correct_predict_result.append(1 if original_label[i]==prediction_run[0] else 0)

print('Final Accuracy:', str(np.mean(correct_predict_result) * 100) + '%')

