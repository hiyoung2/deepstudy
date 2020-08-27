# part1 - building the CNN

# importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# initialising the CNN (CNN 초기화==선언)
model = Sequential()

# step1 - convolution
model.add(Conv2D(32,(3,3), input_shape = (100, 100, 3), activation='relu'))

# step2 - pooling
model.add(MaxPooling2D(pool_size = (2,2)))

# Adding a second convolutional layer
model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# step3 - flattening
model.add(Flatten())

# step4 - Full connection
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# part2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_set = train_datagen.flow_from_directory(
    './project/cnn/data/train', target_size=(100,100), batch_size=25, class_mode='binary')

test_set = test_datagen.flow_from_directory(
    './project/cnn/data/test', target_size=(100,100), batch_size=25, class_mode='binary')

model.fit_generator(
    train_set, steps_per_epoch = 8, epochs= 32, 
    validation_data = test_set, validation_steps = 4)


# steps_per_epoch = traindata수/batchsize
# validation_steps = validationdata수/batchsize


output = model.predict_generator(test_set, steps=15)
print(test_set.class_indices)
print(output)
print("완료")