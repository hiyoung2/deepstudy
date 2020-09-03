from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Flatten, Dropout, Dense, Input
from keras.preprocessing.image import ImageDataGenerator

def callbacks(model_path, patience):
    callbacks = [
        ReduceLROnPlateau(patience=3),  # TODO Change to cyclic LR
        ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
        EarlyStopping(patience=patience)  # EarlyStopping needs to be placed last, due to a bug fixed in tf2.2
    ]
    return callbacks


def frozen_resnet(input_size, n_classes):
    model_ = ResNet50V2(include_top=False, input_tensor=Input(shape=input_size))
    for layer in model_.layers:
        layer.trainable = False 
    x = Flatten(input_shape=model_.output_shape[1:])(model_.layers[-1].output)
    x = Dense(n_classes, activation='softmax')(x)
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
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['acc']
)

steps_per_epoch_train = int((1620) / batch_size)
model_path_finetune = 'model_finetuned.h5'

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    validation_split= 0.2)

dataset_path = 'D:/python_module/darknet-master/build/darknet/x64/project/flow_from'

train_gen = datagen.flow_from_directory(
    directory=dataset_path,
    batch_size=batch_size,
    target_size=input_size[:-1],
    shuffle=True,
    subset='training')

val_gen = datagen.flow_from_directory(
    directory=dataset_path,
    batch_size=batch_size,
    target_size=input_size[:-1],
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





