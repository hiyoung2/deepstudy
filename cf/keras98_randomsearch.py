# 정식명 : RandomizedSearchCV
# 97번을 RandomizedSearchCv 로 변경하시오

# score를 넣어주자

from keras.datasets import mnist
from keras.utils import np_utils # 사이킷런의 원핫인코딩 써도 됨
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten
from keras.layers import Dense, MaxPooling2D
import numpy as np


# keras에서 데이터 불러오는 방법
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train.shape :", x_train.shape) # (60000, 28, 28)
print("x_test.shape : ", x_test.shape)  # (10000, 28, 28)
print("y_train.shape :", y_train.shape) # (60000,)
print("y_test.shape :", y_test.shape)   # (10000,)


# 1. 데이터

# astype('float32') 안 써도 됨
# 0 ~ 255 , 255로 나누면 MinMaxScaler와 동일한 효과

###########################################################################################
# for Conv2D(4차원)
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)/255
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)/255

# CNN : 4차원! 따라서 6000, 28, 28, 1 로 reshape 해 준다
# print("x_train.re :",x_train.shape)  # (60000, 28, 28, 1)
# print("x_test.re :",x_test.shape)    # (10000, 28, 28, 1)

###########################################################################################
# for Dense(2차원)
x_train = x_train.reshape(x_train.shape[0], 28*28)/255
x_test = x_test.reshape(x_test.shape[0], 28*28)/255



# 분류모델, 당연히 one-hot encoding 해줘야함
# 케라스에는 np_utils에서 제공하는 to_categorical 원핫인코딩 방법을 한다
# 라벨이 0부터 시작한다

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print("y_train.shape.oh :", y_train.shape) # (60000, 10)
print("y_test.shape.oh :", y_test.shape)   # (10000, 10)


# 2. 모델 구성

# 모델 자체를 진짜 함수로 만든다(어렵지 않다)
# 함수 : 재사용가능!

# build_model 이란 모델을 만들겠고, 파라미터 : drop, optimizer 사용

# for 문을 쓰면 이 모델을 여러 번 돌릴 수도 있다

# GridSearch / RandomizedSearch 에 필요한 모델, 파라미터 모두 준비 되었음
# model = GridSearchCV(SVC(), parameters, cv = kfold)

def build_model(drop=0.5, optimizer = 'adam') :
    inputs = Input(shape = (28*28, ), name = 'inputs')
    x = Dense(512, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation = 'softmax', name = 'outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics = ['acc'],
                  loss = 'categorical_crossentropy')
    return model

# 현재 compile 까지만 들어가 있음
# fit은 어디서? -> GridSearch or RandomizedSearch에서 할 것

# 하피퍼파라미터들도 함수로 작성
def create_hyperparameters() :
    batchs = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5) # 0.1 ~ 0.5 를 5등분, 0.1, 0.2, 0.3, 0.4, 0.5
    return {"batch_size" : batchs, "optimizer" : optimizers, 
            "drop" : dropout} # dictionary 형태

# epoch, node, activation 등등의 파라미터들도 넣을 수 있다
# 512, 256, 128이 들어간 자리에 변수명을 써 주고 node의 갯수 또한 하이퍼파라미터에서 명시를 해서 여러가지 경우의 수를 설정할 수 있다
# 레이어는 조금 복잡해질 수 있다


# 싸이킷런 위에다 케라스 올리고 하이퍼파라미터 올리고 CV 올려서 땡겨 온다

# wrap : 싸다. 싸이킷런을 불러서 wrapping 하겠다
# 우리는 분류모델(또는 회귀모델)에 대한 싸이킷런 형식의 wrapping을 하겠다
# keras를 wrapping 해서 scikit-learn을 사용
# 케라스 모델을 싸이킷런 작업의 일부로 사용

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

# model을 싸이킷런에서 쓸 수 있게 wrapping 했다
# 아래 모델 준비는 싸이킷런에서 사용하는 방식
# wrapping을 했기 때문에 이렇게 쓸 수 있다
model = KerasClassifier(build_fn = build_model, verbose = 1) # 모델 준비
                       # 직접 만든 함수형 모델   
hyperparameters = create_hyperparameters() # 하이퍼파라미터 준비
                       # 직접 만든 하이퍼파라미터들
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search = RandomizedSearchCV(model, hyperparameters, cv = 3)

search.fit(x_train, y_train)
acc = search.score(x_test, y_test)

print("최적의 파라미터 :", search.best_params_)
# search.best_prams_ : search.best_estimator_ 와 거의 같다

print("acc :", acc)

'''
score 구하지 않았을 때 (1차 실행)
최적의 파라미터 : {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 40}

score 구할 때 (2차 실행)
최적의 파라미터 : {'optimizer': 'adadelta', 'drop': 0.1, 'batch_size': 40}
acc : 0.9677000045776367

# score만 추가해서 돌렸는데 또 랜덤으로 나옴
optimizer가 adam -> adadelta 

3차 실행

'''


# 케라스 gpu
# 그리드서치 cpu
# cpu만 돌릴 때는 n_jobs 써도 괜찮은데
# gpu랑 같이 쓸 때는 안 쓰는 게 좋겠다, 자꾸 터짐