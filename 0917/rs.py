# import libraries
import os
import glob
import cv2
import tensorflow as tf
import numpy as np

from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam
from keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split as tts
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV


# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()
# with tf.device('/gpu:0'): sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))


print("import complete")

def load_train(train_path, image_size, classes):
    images = []
    labels = []

    print("Going to read images for training")
    # classes = ["fighting", "normal"]
    for fields in classes:
        index = classes.index(fields)
        print(f"Reading {fields} files index: {index}")
        path = os.path.join(train_path, fields, '*jpg')
        print(path)

        # make a file list
        files = glob.glob(path)
        for file in files:
            # dowin-sizing images
            image = cv2.imread(file, 0)
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_AREA)

            # apply MinMaxScaler to images(x data / input data)
            image = image.astype(np.float32) / 255.0
            image_n = image.reshape(224, 224, 1)
            images.append(image_n)

            # y data(output data)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


train_path = "./0917/data"
image_size = 224
classes = ["fighting", "normal"]

data = load_train(train_path, image_size, classes)

print(f"x:\n{data[0]}")
print()
print(f"y:\n{data[1]}")
print()

# preparing data
x = data[0]
y = data[1]

x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.2, random_state = 77)

print("x_train.shape :", x_train.shape) # (1600, 224, 224, 1)
print("y_train.shape :", y_train.shape) # (1600, 2)
print("x_test.shape :", x_test.shape) # (400, 224, 224, 1)
print("y_test.shape :", y_test.shape) # (400, 2)

# define convolutional neural network
def build_model(dropout = 0.1, optimizer = 'adam', learning_rate = 0.1, activation = 'relu', kernel_size = (2, 2)):
    inputs = tf.keras.Input(shape = (224, 224, 1), name = 'inpupts')
    x = tf.keras.layers.Conv2D(32, kernel_size = kernel_size, padding = 'same', activation = activation)(inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size = kernel_size, padding = 'same', activation = activation)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size = kernel_size, strides = 2)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size = kernel_size, padding = 'same', activation = activation)(x)
    x = tf.keras.layers.Conv2D(256, kernel_size = kernel_size, padding = 'same', activation = activation)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16)(x)
    outputs = tf.keras.layers.Dense(2, activation = 'sigmoid')(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    
    model.compile(optimizer = optimizer(learning_rate = learning_rate), loss = 'binary_crossentropy', metrics = ['acc'])
    
    return model



def create_hyperparameters():

    batches = [8, 32, 64, 128, 256]
    optimizers = [RMSprop, Adam, Adadelta, Adagrad, SGD, Nadam]
    lr = [1e-4, 1e-3, 1e-2]
    dropout = [0.1, 0.2, 0.3, 0.4]
    activation = ['relu', 'elu', 'selu', 'tanh', 'sigmoid', LeakyReLU()]
    kernel_size = [2, 3, 4]
    epochs = [64, 128, 256, 512]

    return {"batch_size": batches, "optimizer": optimizers, "learning_rate": lr,
            "dropout": dropout, "activation": activation, "kernel_size": kernel_size, "epochs": epochs}


model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_model, verbose=1)

hyperparams = create_hyperparameters()

rs = RandomizedSearchCV(model, hyperparams, cv=3)

rs.fit(x_train, y_train)

print(f"best_params :\b=n {rs.best_params_}")

acc = rs.score(x_test, y_test)
print("accuracy :", acc)

    