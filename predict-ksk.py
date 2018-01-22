import os
import sys
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
import numpy as np
import shutil
from datetime import datetime

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
result_dir = 'results'
model_name = "finetuning-ksk.h5"
#TestDir = "/media/deepstation/df661f44-30ab-4e78-be4b-85a6014ac61d/deepstation/prj_data/ai-ocr/classification"
TestDir = "/media/deepstation/df661f44-30ab-4e78-be4b-85a6014ac61d/deepstation/prj_data/ai-ocr/gt"
IS_NEED_2_COPY = False

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
   # 正規化
   x = x / 255.0
   # クラスを予測
   pred = model.predict(x)[0]
   # top 1
   top = 1
   top_indices = pred.argsort()[-top:][::-1]
   result = [(classes[i]) for i in top_indices]
   cls = result[0]

   return cls

def getPredictTop5 (filePath):
   filename = filePath
   img = image.load_img(filename, target_size=(img_height, img_width))
   x = image.img_to_array(img)
   x = np.expand_dims(x, axis=0)
   x = x / 255.0
   pred = model.predict(x)[0]
   top = 1
   top_indices = pred.argsort()[-top:][::-1]
   result = [(classes[i], pred[i]) for i in top_indices]
   result = result[0]
   result = str(result[0]) + "," + str(result[1])
   return result

print("=============START========================")
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
print(os.path.join(result_dir, model_name))
model.load_weights(os.path.join(result_dir, model_name))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# model.summary()

if not os.path.exists(TestDir):
  print("bad testdir")
  exit()
"""
save result to csv files
"""
parDirPath = os.path.abspath(os.path.join(TestDir, os.pardir))
PreDir = os.path.join(parDirPath,"predict")
print(PreDir)
if os.path.exists(PreDir):
  print("Rename PreDir")
  dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  os.rename(PreDir, PreDir + "-bk" + dt)
createDir(PreDir)
for i in range(len(classes)):
   dirPath = os.path.join(PreDir, classes[i])
   createDir(dirPath)
ResultDir = os.path.join(TestDir, 'result.csv')
files = [os.path.join(root, name)
             for root, dirs, files in os.walk(TestDir)
             for name in files
             if name.endswith((".jpg", ".png"))]
resultList = []
mis = 0
for i in range(len(files)):
  print(i)
  filePath = files[i] #os.path.join(TestDir, files[i])
  predResult = getPredictTop5(filePath)
  gt = filePath.split('/')[-2]
  re = predResult.split(',')[0]
  predCls = getPredictCls(filePath)
  if IS_NEED_2_COPY:
    dstPath = os.path.join(PreDir, predCls)
    dstPath = os.path.join(dstPath, os.path.basename(files[i]))
    try:
       shutil.copyfile(filePath, dstPath)
       print(dstPath)
    except shutil.SameFileError:
       print("the same file")

  if not gt == re:
    mis += 1
    result = (filePath.split('/')[-2]) + "," + (filePath.split('/')[-1]) + "," + str(predResult)
    resultList.append(result)

f = open(ResultDir, 'w')
for i in range(len(resultList)):
   f.write(resultList[i] + "\n")
f.close()

accuracy = 1 - (float)(mis/(len(files)))
print("Accuracy: " + str(accuracy))

print("=============END========================")
