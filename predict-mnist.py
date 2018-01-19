import os
import sys
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
import numpy as np
import shutil

result_dir = 'results'

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
nb_classes = len(classes)

img_height, img_width = 56, 56
channels = 3

def createDir(dirPath):
  if not os.path.exists(dirPath):
    os.mkdir(dirPath)

def getPredictCls(filePath):
   filename = filePath
   # 画像を読み込んで4次元テンソルへ変換
   img = image.load_img(filename, target_size=(img_height, img_width))
   x = image.img_to_array(img)
   x = np.expand_dims(x, axis=0)

   # 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要！
   # これを忘れると結果がおかしくなるので注意
   x = x / 255.0
   # クラスを予測
   # 入力は1枚の画像なので[0]のみ
   pred = model.predict(x)[0]

   # 予測確率が高いトップnを出力
   top = 1
   top_indices = pred.argsort()[-top:][::-1]
   #result = [(classes[i], pred[i]) for i in top_indices]
   result = [(classes[i]) for i in top_indices]
   cls = result[0]

   return cls

# VGG16
input_tensor = Input(shape=(img_height, img_width, channels))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC
fc = Sequential()
fc.add(Flatten(input_shape=vgg16.output_shape[1:]))
fc.add(Dense(256, activation='relu'))
fc.add(Dropout(0.5))
fc.add(Dense(nb_classes, activation='softmax'))

# VGG16とFCを接続
model = Model(input=vgg16.input, output=fc(vgg16.output))

# 学習済みの重みをロード
model.load_weights(os.path.join(result_dir, 'finetuning-mnist.h5'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# model.summary()

TestDir = "/media/deepstation/df661f44-30ab-4e78-be4b-85a6014ac61d/deepstation/1j-ocr-20171221/cut"
#TestDir = "/media/deepstation/df661f44-30ab-4e78-be4b-85a6014ac61d/deepstation/ocr-kskdata-20171222/test-train"
if not os.path.exists(TestDir):
  print("bad testdir")
  exit()
parDirPath = os.path.abspath(os.path.join(TestDir, os.pardir))
PreDir = os.path.join(parDirPath,"predict")
createDir(PreDir)
for i in range(len(classes)):
   dirPath = os.path.join(PreDir, classes[i])
   createDir(dirPath)

#files = os.listdir(TestDir)
files = [os.path.join(root, name)
             for root, dirs, files in os.walk(TestDir)
             for name in files
             if name.endswith((".jpg", ".png"))]
print(len(files))
for i in range(len(files)):
  filePath = files[i] #os.path.join(TestDir, files[i])
  predCls = getPredictCls(filePath)
  dstPath = os.path.join(PreDir, predCls)
  dstPath = os.path.join(dstPath, os.path.basename(files[i]))
  try:
     shutil.copyfile(filePath, dstPath)
     print(dstPath)
  except shutil.SameFileError:
     print("the same file")
