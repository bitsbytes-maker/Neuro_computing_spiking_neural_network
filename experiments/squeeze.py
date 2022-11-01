from tensorflow.python.keras.layers import Conv2D, AveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import itertools
from snnbuilder.bin.utils import summarize_diagnostics
import time
import numpy as np


batch_size = 32
training_augs = {}
# training_augs = {'width_shift_range': 0.1,
#                       'height_shift_range': 0.1,
#                       'horizontal_flip': True,
#                       'rotation_range': 15,
#                       'shear_range': 0.2,
#                       'zoom_range': 0.2,
#                       'fill_mode': 'nearest',
#                       }
datagen_kwargs = {'data_format': 'channels_last', 'rescale': 1.0 / 255.0}
dataflow_kwargs = {'class_mode': 'categorical', 'target_size': (224, 224)}
epochs = 50


def load_dataset(data_path):
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')

    # check dataset exists
    if not os.path.isdir(data_path):
        raise NotADirectoryError

    train_datagen = ImageDataGenerator(**datagen_kwargs, validation_split=0.1, **training_augs)
    train_iter = train_datagen.flow_from_directory(train_dir, **dataflow_kwargs,
                                                   batch_size=batch_size, subset='training',
                                                   shuffle=True)

    # no image augmentation on validation or test set
    # ImageDataGenerator validation_split reserves the last n% of data, before any shuffling, so validation split
    # is stable
    val_datagen = ImageDataGenerator(**datagen_kwargs, validation_split=0.1)
    val_iter = val_datagen.flow_from_directory(train_dir, **dataflow_kwargs, batch_size=batch_size,
                                               subset='validation', shuffle=True)

    test_datagen = ImageDataGenerator(**datagen_kwargs)
    test_iter = test_datagen.flow_from_directory(val_dir, **dataflow_kwargs, batch_size=batch_size,
                                                 shuffle=True)

    return train_iter, val_iter, test_iter


def load_dataset_numpy(data_path):  # todo eats memory
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'val')

    datagen = ImageDataGenerator(**datagen_kwargs, dtype=np.float32)
    data_iter = datagen.flow_from_directory(train_dir, **dataflow_kwargs, batch_size=batch_size)

    train_x = np.zeros([data_iter.samples] + list(data_iter.image_shape), dtype=np.float32)
    train_y = np.zeros([data_iter.samples] + [len(data_iter.class_indices)], dtype=np.float32)
    for i in range(0, data_iter.samples, data_iter.batch_size):
        batch = data_iter.next()
        b = len(batch[0])
        train_x[i:i + b, 0:224, 0:224, 0:3] = batch[0]
        train_y[i:i + b, 0:2] = batch[1]

    datagen = ImageDataGenerator(**datagen_kwargs, dtype=np.float32)
    data_iter = datagen.flow_from_directory(test_dir, **dataflow_kwargs, batch_size=batch_size)

    test_x = np.zeros([data_iter.samples] + list(data_iter.image_shape), dtype=np.float32)
    test_y = np.zeros([data_iter.samples] + [len(data_iter.class_indices)], dtype=np.float32)
    for i in range(0, data_iter.samples, data_iter.batch_size):
        batch = data_iter.next()
        b = len(batch[0])
        test_x[i:i + b, 0:224, 0:224, 0:3] = batch[0]
        test_y[i:i + b, 0:2] = batch[1]

    return (train_x, train_y), (test_x, test_y)


def fire_module(x, squeeze_filters=3, ki='he_normal'):
    expand_filters = 4 * squeeze_filters
    squeeze = Conv2D(filters=squeeze_filters, kernel_size=(1, 1), padding='same', strides=(1, 1), activation='relu', kernel_initializer=ki)(x)
    # changed e1 to use 'same' padding instead of 'valid' to fix kernel size issue with inputs
    e1 = Conv2D(filters=expand_filters, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer=ki)(squeeze)
    e3 = Conv2D(filters=expand_filters, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=ki)(squeeze)
    concat = layers.Concatenate(axis=-1)([e1, e3])
    return concat


def model(input_shape=(224,224,3), ki='he_normal'):
    num_classes = 2

    input_layer = keras.Input(shape=input_shape, name='input_layer')
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation="relu",
                      kernel_initializer=ki)(
        input_layer)
    x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x = fire_module(x, squeeze_filters=16, ki=ki)
    x = fire_module(x, squeeze_filters=16, ki=ki)
    x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x = fire_module(x, squeeze_filters=32, ki=ki)
    x = fire_module(x, squeeze_filters=32, ki=ki)
    x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x = fire_module(x, squeeze_filters=48, ki=ki)
    x = fire_module(x, squeeze_filters=48, ki=ki)
    x = fire_module(x, squeeze_filters=64, ki=ki)
    x = fire_module(x, squeeze_filters=64, ki=ki)

    x = Dropout(0.3)(x)
    x = layers.Conv2D(filters=num_classes, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(input_layer, x, name='squeeze_opt')
    return model


def run(data_path, **kwargs):
    train_iter, val_iter, test_iter = load_dataset(data_path)
    optimizer = kwargs.get('optimizer')
    ki = kwargs.get('ki')
    m = model(ki=ki)

    # custom learning rate schedule
    def scheduler(epoch, lr):
        min_lr = 0.0001
        max_lr = 0.001
        n_cycles = 2

        if (epoch // n_cycles) % 2 == 0:
            return min_lr
        else:
            return max_lr

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_schedule = keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

    m.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = m.fit(train_iter, epochs=epochs, validation_data=val_iter, verbose=2, callbacks=[early_stopping, lr_schedule])
    _, accuracy = m.evaluate(test_iter, verbose=2)

    return accuracy, history


def run_numpy(train, test, **kwargs):
    optimizer = kwargs.get('optimizer')
    ki = kwargs.get('ki')
    m = model(ki=ki)

    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    m.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = m.fit(train, epochs=epochs, validation_split=0.1, shuffle=True, verbose=2, callbacks=[callback])
    _, accuracy = m.evaluate(test, verbose=2)

    return accuracy, history


def main(output_path=None, data_path=None):
    model_name = 'squeeze_opt'

    # output_path = '/home/patrick/Documents/snntoolbox_outputs'
    # data_path = '/media/patrick/HDD/VLSI_VOC/neuro_comp/dev_outputs/datasets' + '/catdog'

    if output_path is None:
        output_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../outputs'))
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

    path_parent = os.path.abspath(os.path.join(output_path, model_name))
    if not os.path.isdir(path_parent):
        os.makedirs(path_parent)

    if data_path is None:
        data_path = os.path.join(path_parent, '../datasets/catdog')
        if not os.path.isdir(data_path):
            os.makedirs(data_path)

    params = {'optimizer': [Adam(), SGD(momentum=0.9, nesterov=True)],
              'ki': ['he_normal', 'he_uniform']
              }
    keys = list(params.keys())

    logfile = os.path.join(path_parent, 'exp_log.txt')
    with open(logfile, 'w', encoding='utf-8') as f:
        f.write('optimizing {}\n\n'.format(keys))

    for values in itertools.product(*params.values()):
        d = {keys[i]: values[i] for i in range(len(values))}
        print('$' * 30)
        print([keys[i] + ': ' + str(values[i]) for i in range(len(values))])
        print('$' * 30)
        accuracy, history = run(data_path, **d)

        with open(logfile, 'a', encoding='utf-8') as f:
            line = str([keys[i] + ': ' + str(values[i]) for i in range(len(values))]) + '\n\n'
            f.write('accuracy: {}\n'.format(accuracy))
            f.write(line)


def ln(output_path=None, data_path=None):
    model_name = 'squeeze_opt'

    output_path = '/home/patrick/Documents/snntoolbox_outputs'
    data_path = '/media/patrick/HDD/VLSI_VOC/neuro_comp/dev_outputs/datasets' + '/catdog'

    if output_path is None:
        output_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../outputs'))
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

    path_parent = os.path.abspath(os.path.join(output_path, model_name))
    if not os.path.isdir(path_parent):
        os.makedirs(path_parent)

    path_wd = os.path.abspath(os.path.join(path_parent, str(time.time())))
    os.makedirs(path_wd)

    if data_path is None:
        data_path = os.path.join(path_parent, '../datasets/catdog')
        if not os.path.isdir(data_path):
            os.makedirs(data_path)

    # todo think i 100% need a learning schedule for squeeze, start small, maybe get larger

    # todo maybe should go even lower with learning rate for SGD?
    # also try he_uniform for best learning rate
    # also try learning schedule, 1cycle with n1 picked out
    # also maybe RMS prop with learnign rate variable or something

    # params = {'optimizer': [Adam(), SGD(momentum=0.9, nesterov=True)],
    #           'ki': ['he_normal', 'he_uniform']
    #           }
    # todo lr of 1.1 i think is too high, just get nan for loss
    # params = {'ln': [0.1, 0.01, 0.001]}
    # params = {'ln': [0.001]}
    params = {
              'nesterov': [True, False]
              }

    keys = list(params.keys())

    logfile = os.path.join(path_wd, 'exp_log.txt')
    with open(logfile, 'w', encoding='utf-8') as f:
        f.write('optimizing {}\n\n'.format(keys))

    trial_number = 0
    # train, test = load_dataset_numpy(data_path)

    for values in itertools.product(*params.values()):
        d = {keys[i]: values[i] for i in range(len(values))}
        print('$' * 30)
        print([keys[i] + ': ' + str(values[i]) for i in range(len(values))])
        print('$' * 30)

        optimizer = SGD(momentum=0.9, nesterov=d['nesterov'], learning_rate=0.001)
        accuracy, history = run(data_path, **d, optimizer=optimizer, ki='he_normal')
        # accuracy, history = run_numpy(train, test, **d, optimizer=optimizer, ki='he_normal')

        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        epochs_trained = len(history.epoch)

        summarize_diagnostics(history, path_wd, str(trial_number))

        with open(logfile, 'a', encoding='utf-8') as f:
            f.write('trial {}\n'.format(trial_number))
            line = str([keys[i] + ': ' + str(values[i]) for i in range(len(values))]) + '\n'
            f.write(line)
            f.write('train accuracy: {:.2f}\n'.format(train_acc))
            f.write('val accuracy: {:.2f}\n'.format(val_acc))
            f.write('test accuracy: {:.2f}\n'.format(accuracy))
            f.write('epochs: {}\n'.format(epochs_trained))
            f.write('opt hparams: {}\n'.format(optimizer._hyper))
            f.write('\n')

        trial_number += 1


if __name__ == '__main__':
    ln()
    # main()

