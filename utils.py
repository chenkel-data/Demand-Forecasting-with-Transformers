import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


def univariate_data(dataset, start_index, end_index, history_size, target_size, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
            end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))

        if single_step:
            labels.append(dataset[i+target_size])
        else:
            labels.append(dataset[i:i+target_size])


    return np.array(data), np.array(labels)


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

  #  print(dataset)
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    print(start_index,end_index)
    for i in range(start_index, end_index):
        if single_step:
            labels.append(target[i+target_size])
        else:
            if len(np.asarray(target[i:i+target_size]))==target_size:
                indices = range(i-history_size, i, step)
                data.append(dataset[indices])

                labels.append(np.asarray(target[i:i+target_size], np.float32))

    return np.array(data), np.array(labels)

#plot functions
def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']

    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta

    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
            label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt


def multi_step_plot(history, true_future, prediction):
    STEP=1
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 0]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), '--bo',
           label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), '--ro',
             label='Predicted Future')
    plt.legend(loc='upper left')

    #y_max=max(max(max(true_future), max(prediction)), max(history[:, 0]))
    #y_min=min(min(min(true_future), min(prediction)),  min(history[:, 0]))
    #plt.ylim(y_min-0.5,y_max+0.5)
    plt.show()

def create_time_steps(length):
    return list(range(-length, 0))

#plot loss
def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()
