# -*- coding: utf-8 -*-
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image

class VGG_OCR:

  def __init__(self, classes, model_path, img_height=56, img_width=56, channels=3):
    self.classes = classes
    self.img_height = img_height
    self.img_width = img_width
    nb_classes = len(classes)
    # VGG16
    input_tensor = Input(shape=(img_height, img_width, channels))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
  
    # FC
    fc = Sequential()
    fc.add(Flatten(input_shape=vgg16.output_shape[1:]))
    fc.add(Dense(256, activation='relu'))
    fc.add(Dropout(0.5))
    fc.add(Dense(nb_classes, activation='softmax'))
  
    # link FC to VGG
    self.model = Model(input=vgg16.input, output=fc(vgg16.output))
  
    # load weight file
    self.model.load_weights(model_path)
  
    # compile
    self.model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

  def classify(self, filePath):
     img = image.load_img(filePath, target_size=(self.img_height, self.img_width))
     x = image.img_to_array(img)
     x = np.expand_dims(x, axis=0)
     x = x / 255.0
     pred = self.model.predict(x)[0]
     top = 1
     top_indices = pred.argsort()[-top:][::-1]
     result = [(self.classes[i], pred[i]) for i in top_indices]
     result = result[0]
     className = str(result[0])
     confidence = float(result[1])
     return className, confidence

  
if __name__ == "__main__":
  print("=========start=======")
  classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
  model_path = "results/finetuning-ksk.h5"
  vgg_ocr = VGG_OCR(classes, model_path)

  filePath = "test.jpg"
  cls, conf = vgg_ocr.classify(filePath)
  print(str(cls) + ", " + str(conf))
  print("=========end=========")