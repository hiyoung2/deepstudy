from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Flatten, Dropout, Dense, Input
from keras.preprocessing.image import ImageDataGenerator

def callbacks(model_path, patience):
    callbacks = [
        ReduceLROnPlateau(patience=3),  # TODO Change to cyclic LR
        # ReduceLROnPlateau 
        # keras의 callback 함수, 학습률이 개선되지 않을 때, 학습률을 동적으로 저정, 학습률을 개선하는 효과 기대
        # 경사하강법에 의해 학습을 하는 경우, local minima에 빠져버리게 되면, 더이상 학습률 개선X or 정체 or 심하게 튀는 현상 발생
        # Local Minima에 빠져버린 경우, 쉽게 빠져나오지 못하고 갇혀버리게 됨, 이 때 learning rate를 늘리거나 줄여주는 방법으로 빠져나오는 효과 기대 가능
        # Parameteres
        # 1) monitor : 검증 손실 기준 ex)val_loss, val loss 개선 되지 않으면 callback 호출
        # 2) factor : 0.5 라고 하면, callback 호출 시 학습률을 1/2로 줄인다
        # 3) verbose : 0 - quiet, 1 - update messages
        # 4) mode = auto, min, max / min : lr 감소, monitor 기준 감소시에 / max : lr 감소, monitor 기준 증가시에 // auto 알아서(default)
        # 5) min_delta : threshold for measuring the new optimum, to only focus on significant changes
        # 6) cooldown : number of epochs to wait before resuming normal operation after lr has been reduced.
        # 7) min_lr : lower bound on the learning rate.

        # example
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
        #                         patience=5, min_lr=0.001)
        # model.fit(X_train, Y_train, callbacks=[reduce_lr])

        

        # model 의 weight 저장
        ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
        # model_path : 사용 모델
        # monitor : 'val loss'를 기준으로 본다
        # save_best_only : 모델의 monitor 기준이 가장 좋을 때만 저장하도록 하는 옵션


        EarlyStopping(patience=patience)  # EarlyStopping needs to be placed last, due to a bug fixed in tf2.2
    ]
    return callbacks


# finetuning case 1 (Conv는 Frozen, Fc layer는 train)
def frozen_resnet(input_size, n_classes):
    model_ = ResNet50V2(include_top=False, input_tensor=Input(shape=input_size))
    # include_top = False : flatten() layer 전 층까지만 가져다 쓴다
    # include_top = True : flatten() layer 까지 다 가져 와서 쓴다
    # input_size : 사용할 이미지 크기

    for layer in model_.layers:
        layer.trainable = False # frozen 상태로 가중치만 가져온다
    # x = Flatten(input_shape=model_.output_shape[1:])(model_.layers[-1].output)
    x = Flatten()(model_.layers[-1].output)
    x = Dense(n_classes, activation='softmax')(x)
    # output layer, n_classes : 클래스 개수(라벨의 개수)
    frozen_model = Model(model_.input, x)
    # 함수형 모델이므로 마지막에 input과 output을 Model()에 넣어준다

    return frozen_model


classes = ['normal', 'fighting']
# input_size = (3840,2160,3)
input_size = (416,416,3) # 훈련에 사용할 이미지 크기 설정
n_classes = 2 # 클래스 개수
batch_size = 32 # 배치 사이즈
epochs_finetune = 1 # fine tuning 시 사용할 epoch
epochs_fulltune = 10 # full tuning 시 사용할 epoch


## 모델링 ##
model = frozen_resnet(input_size, n_classes)
# model = frozen_resnet((416, 416, 3), 2)
# 416, 416, 3 : input 
# n_classes : output (normal, fight 2개)


model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['acc']
)

steps_per_epoch_train = int((1620) / batch_size)
# model_path_finetune = 'model_finetuned.h5'
model_path_finetune = './model_finetuned.h5'

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    validation_split= 0.2)

# dataset_path = 'D:/python_module/darknet-master/build/darknet/x64/project/flow_from'
dataset_path = 'D:/deepstudy/project/cnn/data/train'

train_gen = datagen.flow_from_directory(
    directory=dataset_path,
    batch_size=batch_size,
    target_size=input_size[:-1],
    class_mode = 'binary',
    shuffle=True,
    subset='training')

val_gen = datagen.flow_from_directory(
    directory=dataset_path,
    batch_size=batch_size,
    target_size=input_size[:-1],
    class_mode = 'binary',
    shuffle=True,
    subset='validation')

model.fit_generator(generator=train_gen,
                    steps_per_epoch=steps_per_epoch_train,
                    epochs=epochs_finetune,
                    callbacks=callbacks(
                        model_path=model_path_finetune,
                        patience=5),
                    validation_data=val_gen
                    )

model.load_weights(model_path_finetune)

for layer in model.layers:
    layer.trainable = True

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['acc']
)

model_path_full = 'model_full.h5'
model.fit_generator(generator=train_gen,
                    steps_per_epoch=steps_per_epoch_train,
                    epochs=epochs_fulltune,
                    callbacks=callbacks(
                        model_path=model_path_full,
                        patience=10),
                    validation_data=val_gen
                    )

model.load_weights(model_path_full)




# fine tuning #
# resnet 모델 끌고 온 건 frozen, 묶어두고
# 아래에 직접 추가해준 레이어만 훈련시킨다(FC layer만)
# trainable = False 하면 해당 레이어는 훈련을 하지 않음
# frozen, 묶어둔다는 의미는 이미 사전훈련된 가중치를 더 건드리지 않는 것

# resnet : 이미 imagenet이라는 데이터셋으로 사전훈련된 가중치를 제공
# resnet의 가중치를 frozen 상태에서 fit을 한다 == 이 가중치들은 움직이지 않고
# 아래 추가해준 layer ex.flatten, dense 등만 바뀐다
# trainable false한 레이어들은 그 가중치에서 고정,
# true 이면 fit 할 때 epoch 마다 그 레이어들 계속 변동

# false == 기존의 가중치가 변하지 않음을 의미

# fine tuning -> full tuning
# 훈련을 두 번 해 준다

# 사전 훈련된 가중치는 frozen 되어 있는 레이어에 이미 있고
# 마지막에 추가한 layer(FC layer들)는 훈련 때마다 계속 가중치들이 변동 된다

# 사전 훈련된 가중치는 얼려서 업데이트를 하지 않고
# 추가해준 레이어들은 업데이트를 하는 방식

# fine tuning : 사전 훈련된 가중치를 얼려서 훈련
# full tuning : 사전 훈련 가중치 얼리지 않고 통째로 다 훈련시킴

# 위의 코드는 fine tuning fit 한 후, full tuning fit 한 번 더 해 주는 코드
# fit을 2번 해 준 것이다



