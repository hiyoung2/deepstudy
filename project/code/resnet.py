from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Flatten, Dropout, Dense, Input
from keras.preprocessing.image import ImageDataGenerator
import os, shutil
import time
import datetime
import matplotlib.pyplot as plt



def callbacks(model_path, patience):
    callbacks = [
        ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, 
                          patience = 10), 
        ModelCheckpoint(model_path, monitor='val_loss', 
                        verbose=1, save_best_only=True),
        EarlyStopping(patience=patience)]
    return callbacks



def frozen_resnet(input_size, n_classes):
    model_ = ResNet50V2(include_top=False, 
                        input_tensor=Input(shape=input_size))
    for layer in model_.layers:
        layer.trainable = False
    x = Flatten()(model_.layers[-1].output)
    x = Dense(n_classes, activation='sigmoid')(x)
    frozen_model = Model(model_.input, x)

    return frozen_model

classes = ['normal', 'fighting']
# input_size = (3840,2160,3)


input_size = (416,416,3)
n_classes = 2
batch_size = 32
epochs_finetune = 1
epochs_fulltune = 10


## 모델링 ##
model = frozen_resnet(input_size, n_classes)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(),
    metrics=['acc']
)


dataset_path = 'D:/python_module/darknet-master/build/darknet/x64/project/flow_from'



total_image_num = len(os.listdir(dataset_path + '/' + classes[0])) + len(os.listdir(dataset_path + '/' + classes[1])) ## 3905

steps_per_epoch_train = int(total_image_num / batch_size)

model_path_finetune = 'model_finetuned.h5'



## 이미지 제네레이터 ## preprocess_input : resnet에서 제공해주는 전처리함수

## 이미지 제네레이터
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True, zoom_range=0.3,
    width_shift_range=0.3, height_shift_range=0.3,
    validation_split= 0.2
)

train_gen = datagen.flow_from_directory(
    directory=dataset_path, batch_size=batch_size,
    target_size=input_size[:-1], shuffle=True,
    subset='training')

val_gen = datagen.flow_from_directory(
    directory=dataset_path, batch_size=batch_size,
    target_size=input_size[:-1], shuffle=True,
    subset='validation')


start = time.time()


## fozen model fit
model.fit_generator(generator=train_gen, steps_per_epoch=steps_per_epoch_train,
                    epochs=epochs_finetune, 
                    callbacks=callbacks(model_path=model_path_finetune, patience=5),
                    validation_data=val_gen
                    )

model.load_weights(model_path_finetune)

for layer in model.layers:
    layer.trainable = True

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(),
    metrics=['acc']
)

model_path_full = 'model_full.h5'
hist = model.fit_generator(generator=train_gen, steps_per_epoch=steps_per_epoch_train,
                    epochs=epochs_fulltune,
                    callbacks=callbacks(model_path=model_path_full, patience=10),
                    validation_data=val_gen
                    )



model.load_weights(model_path_full)


# 소요 시간
sec = time.time() - start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]

print(times)



# 시각화
plt.figure(figsize = (15, 13))


plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '.', c = 'blue', label = 'loss')         
plt.plot(hist.history['val_loss'], marker = '.', c = 'red', label = 'val_loss')   
plt.grid() 
plt.title('loss')      
plt.ylabel('loss')      
plt.xlabel('epoch')          
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2) 
plt.plot(hist.history['acc'], marker = '.', c = 'blue', label = 'acc')
plt.plot(hist.history['val_acc'], marker = '.', c = 'red', label = 'val_acc')
plt.grid() 
plt.title('acc')      
plt.ylabel('acc')      
plt.xlabel('epoch')          
plt.legend(['acc', 'val_acc']) 
plt.show()  

