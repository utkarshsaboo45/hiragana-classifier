from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.models import model_from_json

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

IMAGE_HT, IMAGE_WT = 28, 28
NO_OF_CLASSES = 71
INPUT_DIMS = (IMAGE_HT, IMAGE_WT, 3)

hiragana_classifier = Sequential()

hiragana_classifier.add(Conv2D(filters = 64, kernel_size = (3, 3), input_shape = INPUT_DIMS))
hiragana_classifier.add(MaxPooling2D(pool_size = (3, 3)))

hiragana_classifier.add(Flatten())

hiragana_classifier.add(Dense(units = 128, activation = 'relu'))
hiragana_classifier.add(Dense(units = NO_OF_CLASSES, activation = 'softmax'))

hiragana_classifier.compile(optimizer = Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

hiragana_classifier.summary()

train_gen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2, 
                                   fill_mode = 'nearest')

test_gen = ImageDataGenerator(rescale = 1./255)

training_set = train_gen.flow_from_directory('splitdataset/HiraganaTrain',
                                                 target_size = (IMAGE_HT, IMAGE_WT), 
                                                 class_mode = 'categorical', 
                                                 batch_size = 64)

test_set = test_gen.flow_from_directory('splitdataset/HiraganaTest', 
                                            target_size = (IMAGE_HT, IMAGE_WT), 
                                            class_mode = 'categorical', 
                                            batch_size = 64)

hiragana_classifier.fit_generator(training_set, 
                         steps_per_epoch = 62309, 
                         epochs = 10, 
                         validation_data = test_set, 
                         validation_steps = 15608)

hiragana_classifier.save('hiragana_classifier2.h5')

# serialize model to JSON
model_json = hiragana_classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
hiragana_classifier.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

#Predicting the class
from keras.preprocessing import image
import numpy as np
from PIL import Image
import PIL.ImageOps

path = 'datasetOrg/HiraganaTest/YU/kanaYU9.jpg'

img = image.load_img(path, target_size=(28, 28))
img = PIL.ImageOps.invert(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = hiragana_classifier.predict(images)
print(classes[0])