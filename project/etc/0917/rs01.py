# import os
# import glob
# import cv2
# import tensorflow as tf
# import numpy as np
# from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam
# from keras.layers import LeakyReLU
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# #from imblearn.pipeline import Pipeline
# from sklearn.model_selection import RandomizedSearchCV
# import dill as pickle
# from keras.utils import plot_model

# def load_train(train_path, image_size, classes):
#     images = []
#     labels = []
#     img_names = []
#     cls = []

#     print('Going to read training images')
#     for fields in classes:
#         index = classes.index(fields)
#         print('Now going to read {} files (Index: {})'.format(fields, index))
#         path = os.path.join(train_path, fields, '*g')
#         files = glob.glob(path)
#         for file in files:
#             image = cv2.imread(file,0)
#             image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
#             #cv2.imwrite('image', image)
#             #cv2.waitKey(0)
#             #cv2.destroyAllWindows() 
#             image = image.astype(np.float32)
#             image = np.multiply(image, 1.0 / 255.0)
#             image1 = image.reshape(224,224,1)
#             images.append(image1)
#             label = np.zeros(len(classes))
#             label[index] = 1.0
#             labels.append(label)
#             flbase = os.path.basename(file)
#             img_names.append(flbase)
#             cls.append(fields)
#     images = np.array(images)
#     labels = np.array(labels)
#     img_names = np.array(img_names)
#     cls = np.array(cls)

#     return images, labels, img_names, cls

# train_path = "D:/project/data/flow_from"
# image_size = 224
# classes = ["fighting", "normal"]

# result = load_train(train_path, image_size, classes)

# print(f"x: {result[0]}")
# print(f"y: {result[1]}")

# x = result[0]
# y = result[1]




# #1-3. train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)#, validation_split=0.2)

# print(f"x_train.shape: {x_train.shape}, x_test.shape: {x_test.shape}") 
# # x_train.shape: (1600, 224, 224), x_test.shape: (400, 224, 224)
# print(f"y_train.shape: {y_train.shape}, y_test.shape: {y_test.shape}")
# # y_train.shape: (1600, 2), y_test.shape: (400, 2)


# #2. cnn network modeling
# def build_model(drop=0.1, optimizer='adam', learning_rate=0.1, activation='relu', kernel_size=(2,2)):
#     inputs = tf.keras.Input(shape=(224,224,1), name='inputs')
#     # x = tf.keras.layers.Reshape((1,224))(inputs)
#     x = tf.keras.layers.Conv2D(32, kernel_size=kernel_size, padding="same", activation=activation)(inputs)
#     x = tf.keras.layers.Dropout(drop)(x)
#     x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
#     x = tf.keras.layers.Conv2D(64, kernel_size=kernel_size, padding="same", activation=activation)(x)
#     x = tf.keras.layers.Dropout(drop)(x)
#     x = tf.keras.layers.MaxPooling2D(pool_size=kernel_size, strides=2)(x)
#     x = tf.keras.layers.Conv2D(128, kernel_size=kernel_size, padding="same", activation=activation)(x)
#     x = tf.keras.layers.Conv2D(256, kernel_size=kernel_size, padding="same", activation=activation)(x)
#     x = tf.keras.layers.Dropout(drop)(x)
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(16)(x)
#     outputs = tf.keras.layers.Dense(2, activation='sigmoid')(x)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     opt = optimizer(learning_rate=learning_rate)
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#     return model

# model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_model, verbose=1)
# plot_model(model)

# #2-1. set pipeline
# # pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
# #pipe = Pipeline([("scaler", StandardScaler()), ('oversample', SMOTE(random_state=12)), ("model", model)])



# #x_train = x_train.reshape(-1*x_train.shape[1],x_train.shape[2])
# #y_train = y_train.reshape(y_train.shape[0]*y_train.shape[1],y_train.shape[2])

# #2-2. set params
# def create_hyperparameters():
    
#     batches = [8,16,24,32,40]
#     optimizers = [RMSprop, Adam, Adadelta, SGD, Adagrad, Nadam]
#     learning_rate = [1e-5, 1e-4, 1e-3, 1e-2]
#     dropout = np.linspace(0.1,0.5,5).tolist()
#     activation = ['tanh', 'relu', 'elu', "selu", "softmax", "sigmoid", LeakyReLU()]
#     kernel_size = [(2,2),(3,3)]

#     # return: {key, value}
#     return{"batch_size": batches, "optimizer":optimizers,
#            "learning_rate": learning_rate, "drop": dropout, 
#            "activation": activation, "kernel_size": kernel_size}

#     '''return{"model__batch_size": batches, "model__optimizer":optimizers,
#            "model__learning_rate": learning_rate, "model__drop": dropout, 
#            "model__activation": activation}'''

# hyperparams = create_hyperparameters()

# #2-3. set RandomizedSearchCV
# search = RandomizedSearchCV(model, hyperparams, cv=3)#, n_jobs=-1)
# # params = search.get_params().keys()
# # print(f"params: {params}")

# #x_train = x_train.reshape(-1*x_train.shape[1],x_train.shape[2])


# search.fit(x_train, y_train)
# # Check the list of available parameters with `estimator.get_params().keys()
# # params: dict_keys(['cv', 'error_score', 'estimator__memory', 'estimator__steps', 
# # 'estimator__verbose', 'estimator__scaler', 'estimator__model', 'estimator__scaler__copy', 
# # 'estimator__scaler__with_mean', 'estimator__scaler__with_std', 'estimator__model__verbose', 
# # 'estimator__model__build_fn', 'estimator', 'iid', 'n_iter', 'n_jobs', 'param_distributions', 
# # 'pre_dispatch', 'random_state', 'refit', 'return_train_score', 'scoring', 'verbose'])`.
# # error solving..

# #for param in search.get_params().keys():
#     #print(f"param:\n{param}")

# print(f"best_params of cnn network:\n {search.best_params_}")
# # best_params of cnn network:
# # {'optimizer': <class 'keras.optimizers.Adagrad'>, 'learning_rate': 0.0012000000000000001, 'kernel_size': (2, 2), 'drop': 0.1, 'batch_size': 32, 'activation': 'relu'}

# acc = search.score(x_test, y_test)
# print(f"acc: {acc}")
# # acc:  0.699999988079071



# #  plt.figure(figsize=(4.2, 4))
# #     for i, comp in enumerate(rbm.components_):
# #         plt.subplot(10, 10, i + 1)
# #         plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
# #                    interpolation='nearest')
# #         plt.xticks(())
# #         plt.yticks(())
# #     plt.suptitle('64 components extracted by RBM', fontsize=16)
# #     plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

# #     plt.show()


# #'''