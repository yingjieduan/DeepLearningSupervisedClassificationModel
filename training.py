import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf

from utility.io_operation import read_dataset, one_hot_encode
from perceptron_network_design import multilayer_perceptron


################################################################
#-------------- I System configuration -------------------------
# Input and Output folders
input_data = "./resource/diabetes.csv"
output_folder = "./output/"
output_model = "./output/DeepLearningModel/model"
output_standardization_scaler = "./output/Standardization/scaler.pkl"

feature_index = 8
label_index = 8

# Training param
learning_rate = 0.9
max_learning_rate = 1
min_learning_rate = 0.01
decay_speed = 40000.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations

training_epochs = 40000
input_vector_size = 8    # input vactor size, the num of neurons in the input layer
num_class = 2            # how many classes

training_accuracy_shreshhold = 0.99
training_testing_total_shreshhold = 0.96

# variable learning rate
variable_learning_rate = tf.placeholder(tf.float32)

# scaling and standardize the input training data
training_data_scaling_function = preprocessing.StandardScaler()
#training_data_scaling_function = preprocessing.RobustScaler() # many outliers()
#training_data_scaling_function = preprocessing.MinMaxScaler()

optimizer_function = tf.train.GradientDescentOptimizer(variable_learning_rate)
#optimizer_function = tf.train.AdamOptimizer(variable_learning_rate)

################################################################
#--------------- II Read Dataset -------------------------------
def split_training_testing_data(input_data):
    # read data
    X, Y, _ = read_dataset(input_data, feature_index, label_index, data_scaler_path = output_standardization_scaler,
                           training_data_scaling_function = training_data_scaling_function)
    # shuffle dataset
    X, Y = shuffle(X, Y, random_state=1)
    #Convert the dataset into train and test part
    train_feature, test_feature, train_label, test_label = train_test_split(X, Y, test_size = 0.2, random_state = 747)
    return train_feature, test_feature, train_label, test_label


################################################################
#--------------- III Training process --------------------------

def train():
    train_x, test_x, train_y, test_y = split_training_testing_data(input_data)
    input_vector_size = train_x.shape[1]
    num_class = train_y.shape[1]

    saver = tf.train.Saver()       # used to save the the result module

    # training system input placehoder
    x = tf.placeholder(tf.float32, [None, input_vector_size])  # placeholder is input: None - how many vectors; n_dim - dimensions of vectors
    y_ = tf.placeholder(tf.float32, [None, num_class])         # label placehoder
    y = multilayer_perceptron(x)              # predicted label

    # accuracy function
    # tf.argmax return the index whose data value is largest
    # tf.equal return bool value
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # cast bool to float and sum and get the mean value
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))  # Cross-entropy loss function: logits- predict result; labels- labels
    training_step = optimizer_function.minimize(loss_function)

    init = tf.global_variables_initializer()  # initialization globally
    sess = tf.Session()
    sess.run(init)

    # histroy diagram
    training_accuracy_history = []
    testing_accuracy_history = []
    loss_history = np.empty(shape=[1], dtype=float)
    testing_loss_history = np.empty(shape=[1], dtype=float)

    for epoch in range(training_epochs):

        # learning rate decay
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-epoch / decay_speed)

        # usually get training data set here and put them into x and y_
        # usually the training data will be iterated for several times, there iterates training_epochs times
        # Training part
        sess.run(training_step, feed_dict={x: train_x, y_: train_y, variable_learning_rate: learning_rate})          # use all training data to do the training at each epoch

        # run with input of training data
        loss, training_accuracy = sess.run([loss_function, accuracy], feed_dict={x: train_x, y_: train_y})
        loss_history = np.append(loss_history, loss)
        training_accuracy_history.append(training_accuracy)

        # run with input of testing data
        testing_loss, testing_accuracy = sess.run([loss_function, accuracy], feed_dict={x: test_x, y_: test_y})
        testing_accuracy_history.append(testing_accuracy)
        testing_loss_history = np.append(testing_loss_history, testing_loss)

        print(str(epoch) + '- LearningRate: {:10.4f}'.format(learning_rate),
              '\t\t TrLoss: {:8.6f}'.format(loss),
              '\t\t TeLoss: {:8.6f}'.format(testing_loss),
              '\t\t TrAccuracy: {:8.6f}'.format(training_accuracy),
              '\t\t TeAccuracy:  {:8.6f}'.format(testing_accuracy))

        if (((training_accuracy > training_accuracy_shreshhold
            and (training_accuracy + testing_accuracy)/2 > training_testing_total_shreshhold))
            or training_accuracy > 0.9999):
            break

    save_path = saver.save(sess, output_model)
    print('Model saved in file:', save_path)

    plt.subplot(2, 2, 1)
    plt.plot([x for x in loss_history[10:] if x < 1])
    plt.title('Training Loss History')
    plt.ylabel('Loss Level')

    plt.subplot(2, 2, 2)
    plt.plot(training_accuracy_history)
    plt.title('Training Accuracy')
    plt.ylabel('Accuracy Rate')

    plt.subplot(2, 2, 3)
    plt.plot([x for x in loss_history[10:] if x < 1])
    plt.title('Testing Loss History')
    plt.ylabel('Loss Level')

    plt.subplot(2, 2, 4)
    plt.plot(testing_accuracy_history)
    plt.title('Testing Accuracy')
    plt.ylabel('Accuracy Rate')
    plt.xlabel('Epoch Number')
    plt.savefig(output_folder + 'TrainingHistory.png')
    # plt.subplot_tool()
    # plt.show()

    # final accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Test Accuracy: ' + str(sess.run(accuracy, feed_dict={x: test_x, y_:test_y})))

    #final mean error
    predict_test_y = sess.run(y, feed_dict={x: test_x})
    predict_volatility = tf.reduce_mean(tf.square(predict_test_y - test_y))
    print('Mean Squared Error: %.4f' % sess.run(predict_volatility))

    sess.close()


if __name__ == '__main__':
    train()