import matplotlib
import tensorflow as tf
import tensorflow_datasets as tfds
import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import *

matplotlib.rcParams['figure.figsize'] = [10, 7]

# Hyperparameters
seed = 22
tf.random.set_seed(seed)
random.seed(a=seed)
hidden_neurons = 128
number_of_hidden_layers = 1
batch_size = 128
learning_rate = 0.001
max_epochs = 50


# Function that plots the errors by epoch graph
def plot_metric(history_data):
    train_metrics = history_data.history['accuracy']
    for t in range(train_metrics.__len__()):
        train_metrics[t] = 1 - train_metrics[t]
    val_metrics = history_data.history['val_' + 'accuracy']
    for t in range(val_metrics.__len__()):
        val_metrics[t] = 1 - val_metrics[t]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation ' + 'errors')
    plt.xlabel("Epochs")
    plt.ylabel('Percentage of errors')
    plt.legend(["train_" + 'errors', 'val_' + 'errors'])
    plt.show()


# Function that print the optimal hyperparameters
def print_optimal():
    print("Optimal Hyperparameters used")
    print("Seed used: ", seed)
    print("batch size: ", batch_size)
    print("Number of hidden layers: ", number_of_hidden_layers)
    print("Hidden layer number of neurons: ", hidden_neurons)
    print("Learning rate: ", learning_rate)
    print("Max epochs: ", max_epochs)
    print("\n")


# Function to plot the confusion matrix
def plot_matrix(matrix):
    plot_confusion_matrix(matrix)
    plt.show()


# Function to normalize the dataset
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


# Function for calculating the accuracy score
def accuracy_score(y_pred, y):
    is_equal = tf.equal(y_pred, y)
    return tf.reduce_mean(tf.cast(is_equal, tf.float32))


# Function to print the per digit accuracy score ant total accuracy
def print_accuracy_by_digit(y_test, test_classes, result_to_print):
    print("Accuracy breakdown by digit:")
    print("---------------------------")
    label_accs = {}
    for label in range(10):
        label_ind = (y_test == label)
        pred_label = test_classes[label_ind]
        label_filled = tf.cast(tf.fill(pred_label.shape[0], label), tf.int64)
        label_accs[accuracy_score(pred_label, label_filled).numpy()] = label
    for key in label_accs:
        print(f"Digit {label_accs[key]}: {key:.3f}")
    print(f"Total accuracy: {result_to_print[1]:.3f}")


def main():
    # Creating the dataset to use
    # Splitting the data into validation data, test data and training data
    (ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True)

    # Preprocessing the training data
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Preprocessing the test data
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # Creating the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(hidden_neurons, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

    # Training the model
    history = model.fit(ds_train, epochs=max_epochs, validation_data=ds_test)

    # Evaluating the model
    print("\n")
    test_result = model.evaluate(ds_test)
    predictions = model.predict(ds_test)
    print("\n")

    # Printing Hyperparameters
    print_optimal()

    # Printing model results, confusion matrix and percentage of errors by epoch
    plot_metric(history)
    test_y = np.concatenate([y for x, y in ds_test], axis=0)
    true_predictions = predictions.argmax(axis=1)
    print_accuracy_by_digit(test_y, true_predictions, test_result)
    result = confusion_matrix(test_y, true_predictions)
    plot_matrix(result)


if __name__ == '__main__':
    main()
