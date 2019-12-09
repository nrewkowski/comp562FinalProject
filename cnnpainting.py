print('start')

#validation_split in model.fit
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras import backend as K
import numpy as np
from keras.preprocessing import image

paintingPath='___paintingdata'
numTrainSamples=100
numValidationSamples=10
numEpochs=10
batchSize=20

if K.image_data_format()=='channels_first':
    inputShape=(3,150,150)
else:
    inputShape=(150,150,3)
#input_shape=(img_width,img_height,3)
dataGenerator=ImageDataGenerator(
        rescale=1.0/255.0,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
paintingGenerator=dataGenerator.flow_from_directory(directory=paintingPath,batch_size=batchSize,class_mode='categorical',
    subset='training')
print("prainting",paintingGenerator.labels)
validation_generator = dataGenerator.flow_from_directory(
    directory=paintingPath,
    batch_size=batchSize,
    class_mode='categorical',
    subset='validation')

cnnModel=Sequential()
cnnModel.add(Conv2D(32,(3,3),input_shape=inputShape))
cnnModel.add(Activation('relu'))
cnnModel.add(MaxPooling2D(pool_size=(2,2)))
cnnModel.summary()

cnnModel.add(Conv2D(32,(3,3)))
cnnModel.add(Activation('relu'))
cnnModel.add(MaxPooling2D(pool_size=(2,2)))

cnnModel.add(Conv2D(64,(3,3)))
cnnModel.add(Activation('relu'))
cnnModel.add(MaxPooling2D(pool_size=(2,2)))

cnnModel.add(Flatten())
cnnModel.add(Dense(64))
cnnModel.add(Activation('relu'))
cnnModel.add(Dropout(0.5))
cnnModel.add(Dense(1))
cnnModel.add(Activation('sigmoid'))

cnnModel.summary()
#cnnModel.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

from googlenet import create_googlenet
model = create_googlenet()


'''cnnModel.fit_generator(paintingGenerator,
                       steps_per_epoch=numTrainSamples//batchSize,
                       epochs=numEpochs,verbose=2,
                       validation_data=validation_generator,
                       validation_steps=numValidationSamples//batchSize)

imageToPredict=image.load_img('___paintingdata/train_1/17506.jpg')
imageToPredict=imageToPredict.img_to_array(imageToPredict)
imageToPredict=np.expand_dims(imageToPredict,axis=0)

result=cnnModel.predict(imageToPredict)
print(result)'''