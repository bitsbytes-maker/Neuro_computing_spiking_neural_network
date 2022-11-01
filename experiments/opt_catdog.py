# import numpy as np
# from sklearn.model_selection import GridSearchCV
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from tensorflow.python.keras.utils import np_utils
# from tensorflow.keras import layers

# from tensorflow import keras
# # Function to create model, required for KerasClassifier
# def create_model():

#     input_layer = keras.Input(shape=(28,28,1))
#     x = layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(input_layer)
#     x = layers.Flatten()(x)
#     x = layers.Dense(150, activation='relu')(x)
#     x = layers.Dense(10, activation='softmax')(x)

#     model = keras.Model(input_layer, x)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
# # load dataset
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# x_train = x_train / 255
# x_test = x_test / 255

# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
# y_train = np_utils.to_categorical(y_train, 10)
# y_test = np_utils.to_categorical(y_test, 10)
# # create model
# model = KerasClassifier(build_fn=create_model, verbose=1)
# # define the grid search parameters
# batch_size = [40, 60]
# epochs = [10, 20]
# param_grid = dict(batch_size=batch_size, epochs=epochs)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3, verbose=3)
# grid_result = grid.fit(x_train, y_train)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))


# exit()


from snnbuilder.models.catsvsdogs import CNN_CatsVsDogs
import sklearn
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import numpy as np

# data setup
data_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../dev_outputs/datasets/catdog'))

datagen_kwargs = {'data_format': 'channels_last', 'rescale': 1.0 / 255.0}
dataflow_kwargs = {'class_mode': 'categorical', 'target_size': (224, 224)}
training_augs = {'width_shift_range': 0.1,
                 'height_shift_range': 0.1,
                 'horizontal_flip': True,
                 'rotation_range': 15,
                 'fill_mode': 'nearest',
                 'shear_range': 0.1,
                 'zoom_range': 0.2
                 }
train_dir = os.path.join(data_path, 'train')
val_dir = os.path.join(data_path, 'val')
batch_size = 32

# train_datagen = ImageDataGenerator(**datagen_kwargs, validation_split=0.1) #, **training_augs)  # todo no augs
# train_iter = train_datagen.flow_from_directory(train_dir, **dataflow_kwargs,
#                                                batch_size=batch_size, subset='training', shuffle=True)
# val_datagen = ImageDataGenerator(**datagen_kwargs, validation_split=0.1)
# val_iter = val_datagen.flow_from_directory(train_dir, **dataflow_kwargs, batch_size=batch_size,
#                                            subset='validation', shuffle=True)
# test_datagen = ImageDataGenerator(**datagen_kwargs)
# test_iter = test_datagen.flow_from_directory(val_dir, **dataflow_kwargs, batch_size=batch_size, shuffle=True)


# using numpy append is awful but good enough for now
# e = train_iter.next()
# train_x = e[0]
# train_y = e[1]
# for i in range(int(train_iter.samples / train_iter.batch_size)):
#     e = train_iter.next()
#     train_x = np.append(train_x, e[0])
#     train_y = np.append(train_y, e[1])
#
#
# e = val_iter.next()
# val_x = e[0]
# val_y = e[1]
# for i in range(int(val_iter.samples / val_iter.batch_size)):
#     e = val_iter.next()
#     val_x = np.append(val_x, e[0])
#     val_y = np.append(val_y, e[1])

datagen = ImageDataGenerator(**datagen_kwargs, dtype=np.float32)
data_iter = datagen.flow_from_directory(train_dir, **dataflow_kwargs, batch_size=batch_size)

train_x = np.zeros([data_iter.samples] + list(data_iter.image_shape), dtype=np.float32)
train_y = np.zeros([data_iter.samples] + [len(data_iter.class_indices)], dtype=np.float32)
for i in range(0, data_iter.samples, data_iter.batch_size):
    batch = data_iter.next()
    b = len(batch[0])
    train_x[i:i + b, 0:224, 0:224, 0:3] = batch[0]
    train_y[i:i + b, 0:2] = batch[1]

    # for j in range(len(batch[0])):
    #     k = i + j
    #     train_x[k:k+b, 0:225, 0:225, 0:4] = batch[0][j]
    #     train_y[k:k+b, 0:3] = batch[1][j]

        # train_x[i+j] = batch[0][j]
        # train_y[i+j] = batch[1][j]



# train_x = np.zeros([train_iter.samples] + list(train_iter.image_shape), dtype=np.float32)
# train_y = np.zeros([train_iter.samples] + [len(train_iter.class_indices)], dtype=np.float32)
# for i in range(0, train_iter.samples + 1, train_iter.batch_size):
#     batch = train_iter.next()
#     for j in range(len(batch)):
#         train_x[i+j] = batch[0][j]
#         train_y[i+j] = batch[1][j]

# val_x = np.zeros([val_iter.samples] + list(val_iter.image_shape), dtype=np.float32)
# val_y = np.zeros([val_iter.samples] + [len(val_iter.class_indices)], dtype=np.float32)
# for i in range(0, val_iter.samples + 1, val_iter.batch_size):
#     batch = val_iter.next()
#     for j in range(len(batch)):
#         val_x[i+j] = batch[0][j]
#         val_y[i+j] = batch[1][j]

print('DATA SET UP COMPLETE')


def create_model(opt='adam', k=5, filters=16, dense=128):
    # keras.backend.clear_session()

    conv_kwargs = {'activation': 'relu', 'padding': 'same', 'kernel_initializer': 'he_uniform'}
    input_shape = 224, 224, 3

    input_layer = keras.Input(shape=input_shape)
    x = layers.Conv2D(filters=filters, kernel_size=k, **conv_kwargs)(input_layer)
    x = layers.AveragePooling2D(pool_size=2, padding='valid')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(filters=filters * 2, kernel_size=k, **conv_kwargs)(x)
    x = layers.AveragePooling2D(pool_size=2, padding='valid')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(filters=filters * 4, kernel_size=k, **conv_kwargs)(x)
    x = layers.AveragePooling2D(pool_size=2, padding='valid')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(filters=filters * 8, kernel_size=k, **conv_kwargs)(x)
    x = layers.AveragePooling2D(pool_size=2, padding='valid')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(dense, activation='relu')(x)
    x = layers.Dense(2, activation='softmax')(x)

    model = keras.Model(input_layer, x, name='opt_catdog')
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# raise Exception('will hang ubuntu rn')

# https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# model setup
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=batch_size, callbacks=[callback])
model = KerasClassifier(build_fn=create_model, verbose=1, epochs=30, batch_size=batch_size, shuffle=True)

# https://stackoverflow.com/questions/31948879/using-explicit-predefined-validation-set-for-grid-search-with-sklearn
# ps = PredefinedSplit(test_fold=val_iter)
# ps = PredefinedSplit(test_fold=(val_x, val_y))

param_grid = {
    # 'k': [3, 5],
    # 'filters': [16, 32],
    # 'opt': ['adam', 'Adagrad'],
    # 'dense': [128, 256, 512]
    'epochs': [5, 10]
}
# grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=ps, verbose=3, n_jobs=1, pre_dispatch=1)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=3, n_jobs=1, pre_dispatch=1)

# grid_result = grid.fit(train_iter, callbacks=[callback], epochs=30, batch_size=batch_size,
#                        steps_per_epoch=18000//batch_size, validation_steps=2000//batch_size)
# grid_result = grid.fit(train_x, train_y, callbacks=[callback], epochs=30, batch_size=batch_size)
grid_result = grid.fit(train_x, train_y)


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
