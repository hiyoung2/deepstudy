# python import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf

# load data
train = pd.read_csv('./dacon/dacon_emnist/data/train.csv')
test = pd.read_csv('./dacon/dacon_emnist/data/test.csv')

# eda
idx = 318
img = train.loc[idx, '0':].values.reshape(28, 28).astype(int)
digit = train.loc[idx, 'digit']
letter = train.loc[idx, 'letter']

plt.title('Index: %i, Digit: %s, Letter: %s'%(idx, digit, letter))
plt.imshow(img)
# plt.show()

# train model
idx = 318
img = train.loc[idx, '0':].values.reshape(28, 28).astype(int)
digit = train.loc[idx, 'digit']
letter = train.loc[idx, 'letter']

plt.title('Index: %i, Digit: %s, Letter: %s'%(idx, digit, letter))
plt.imshow(img)
# plt.show()

# train model

x_train = train.drop(['id', 'digit', 'letter'], axis=1).values
x_train = x_train.reshape(-1, 28, 28, 1)
x_train = x_train/255

# print(x_train[:5, :])



y = train['digit']
y_train = np.zeros((len(y), len(y.unique())))
for i, digit in enumerate(y):
    y_train[i, digit] = 1

def create_cnn_model(x_train):
    inputs = tf.keras.layers.Input(x_train.shape[1:])

    bn = tf.keras.layers.BatchNormalization()(inputs)
    conv = tf.keras.layers.Conv2D(128, kernel_size=5, strides=1, padding='same', activation='relu')(bn)
    bn = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
    pool = tf.keras.layers.MaxPooling2D((2, 2))(conv)

    bn = tf.keras.layers.BatchNormalization()(pool)
    conv = tf.keras.layers.Conv2D(512, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
    bn = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Conv2D(1024, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
    pool = tf.keras.layers.MaxPooling2D((2, 2))(conv)

    flatten = tf.keras.layers.Flatten()(pool)

    bn = tf.keras.layers.BatchNormalization()(flatten)
    # dense = tf.keras.layers.Dense(2048, activation='relu')(bn)
    dense = tf.keras.layers.Dense(1024, activation='relu')(bn)
    dense = tf.keras.layers.Dense(512, activation='relu')(bn)
    dense = tf.keras.layers.Dense(256, activation='relu')(bn)
    dense = tf.keras.layers.Dense(128, activation='relu')(bn)
    

    bn = tf.keras.layers.BatchNormalization()(dense)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(bn)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model


model = create_cnn_model(x_train)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=64)

# predict
x_test = test.drop(['id', 'letter'], axis=1).values
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = x_test/255

submission = pd.read_csv('./dacon/dacon_emnist/data/submission.csv')
submission['digit'] = np.argmax(model.predict(x_test), axis=1)
submission.head()

# submit
submission.to_csv('./dacon/dacon_emnist/submit/0829_03.csv', index=False)


'''
0824_01 : 제출 best (점수 : 0.8284313725)
128
256
512
1024
flatten 
1000
epo 20
Epoch 20/20
2048/2048 [==============================] - 2s 1ms/sample - loss: 0.0167 - accuracy: 0.9961
'''

'''
0824_02 점수 : 0.75
64
128
256
512
flatten
1024
epo 20
Epoch 20/20
2048/2048 [==============================] - 1s 466us/sample - loss: 0.0463 - accuracy: 0.9893
'''

'''
0824_03 점수 : 	0.7401960784
32
64
128
256
flatten
512
1024
epo20
Epoch 20/20
2048/2048 [==============================] - 0s 228us/sample - loss: 0.0481 - accuracy: 0.9824
'''

'''
0825_1 점수 : 0.75
256
256
512
512
flatten
1024
epo20
Epoch 20/20
2048/2048 [==============================] - 2s 809us/sample - loss: 0.0888 - accuracy: 0.9702
'''

'''
0825_2  점수 : 	0.8137254902
128
256
512
1024
flatten
512
epo20
Epoch 20/20
2048/2048 [==============================] - 2s 884us/sample - loss: 0.0140 - accuracy: 0.9976
'''

'''
0825_03 점수 : 0.8039215686
128
256
512
1024
900
epo20
Epoch 20/20
2048/2048 [==============================] - 2s 1ms/sample - loss: 0.0184 - accuracy: 0.9951
'''

'''
0826_01
128
256
512
1024
flatten
1024

epo 20
Epoch 20/20
2048/2048 [==============================] - 2s 1ms/sample - loss: 0.1107 - accuracy: 0.9604
'''

'''
0826_01_1
128
256
512
1024
flatten
1024

epo 15
Epoch 15/15
2048/2048 [==============================] - 2s 1ms/sample - loss: 0.0628 - accuracy: 0.9775
'''

'''
0826_02 점수 : 	0.6176470588
128
256
512
1024
flatten
512

epo 32
Epoch 32/32
2048/2048 [==============================] - 2s 942us/sample - loss: 0.0933 - accuracy: 0.9688

25, 26 : acc 1.0
'''

'''
0826_02_1 점수 : 	0.8333333333	
128
256
512
1024
flatten
512

epo 24
Epoch 24/24
2048/2048 [==============================] - 2s 917us/sample - loss: 0.0148 - accuracy: 0.9961(가장 점수 높은 파일이랑 최종acc 같음)


'''

'''
0826_03
128
256
512
1024
flatten
512

epo 16
Epoch 16/16
2048/2048 [==============================] - 2s 918us/sample - loss: 0.0699 - accuracy: 0.9805
'''

'''
0826_03_1
128
256
512
1024
flatten
1024

epo 16
Epoch 16/16
2048/2048 [==============================] - 2s 1ms/sample - loss: 0.0705 - accuracy: 0.9790
'''

'''
0826_04 점수 : 0.7009803922	
128
256
512
1024
flatten
1000

optimizer = sgd
epo 32

Epoch 32/32
2048/2048 [==============================] - 2s 912us/sample - loss: 0.0027 - accuracy: 1.0000
'''

'''
0827_01 점수 : 	0.8235294118
128
256
512
1024
flatten
1000
epo 32 
Epoch 32/32
2048/2048 [==============================] - 2s 1ms/sample - loss: 0.0049 - accuracy: 0.9980
'''

'''
0827_02 점수 : 	0.8480392157
128
256
512
1024
flatten
512
256
128
epo 32
Epoch 32/32
2048/2048 [==============================] - 2s 818us/sample - loss: 7.4169e-04 - accuracy: 1.0000
'''

'''
0827_03 점수 : 0.862745098	
02와 모델 구성 동일
epo 20
Epoch 20/20
2048/2048 [==============================] - 2s 820us/sample - loss: 0.0018 - accuracy: 1.0000

'''

'''
0828_01 점수 : 	0.7647058824
128
256
512
1024
flatten
512
256
128
64

epo 20
Epoch 20/20
2048/2048 [==============================] - 2s 784us/sample - loss: 0.0531 - accuracy: 0.9863
'''
'''
0828_02 점수 : 0.8382352941
128
256
512
1024
flatten
1024
512
256
128

epo20
Epoch 20/20
2048/2048 [==============================] - 2s 805us/sample - loss: 0.0030 - accuracy: 1.0000
'''

'''
0828_03 점수 : 0.8480392157
128
256
512
1024
flatten
2048
1024
512
256
128

epo20
Epoch 20/20
2048/2048 [==============================] - 2s 802us/sample - loss: 0.0026 - accuracy: 1.0000
'''
'''
0829_01 점수 : 0.87745################################################
128
256
512
1024
flatten
512
256
128
epo 64

Epoch 61/64
2048/2048 [==============================] - 2s 816us/sample - loss: 4.7677e-04 - accuracy: 1.0000
Epoch 62/64
2048/2048 [==============================] - 2s 827us/sample - loss: 5.1921e-04 - accuracy: 1.0000
Epoch 63/64
2048/2048 [==============================] - 2s 830us/sample - loss: 5.7034e-04 - accuracy: 1.0000
Epoch 64/64
2048/2048 [==============================] - 2s 833us/sample - loss: 4.3024e-04 - accuracy: 1.0000
'''
'''
0829_02 점수 : 0.8578431373	
위와 모델 동일
epo 128

Epoch 125/128
2048/2048 [==============================] - 2s 882us/sample - loss: 1.6184e-05 - accuracy: 1.0000
Epoch 126/128
2048/2048 [==============================] - 2s 879us/sample - loss: 1.4242e-05 - accuracy: 1.0000
Epoch 127/128
2048/2048 [==============================] - 2s 873us/sample - loss: 1.4447e-05 - accuracy: 1.0000
Epoch 128/128
2048/2048 [==============================] - 2s 879us/sample - loss: 1.5875e-05 - accuracy: 1.0000
'''

'''
0829_03 점수 : 	0.8823529412
128
256
512
1024
flatten
1024
512
256
128

epo64
'''