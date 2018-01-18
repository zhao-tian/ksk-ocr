import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
from smallcnn import save_history


classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

batch_size = 32
nb_classes = len(classes)

img_rows, img_cols = 56, 56
channels = 3

#DATA_ROOT = "/Volumes/deepstation/ocr-20171221/"
#DATA_ROOT = "/media/deepstation/df661f44-30ab-4e78-be4b-85a6014ac61d/deepstation/ocr-20171221/"
DATA_ROOT = "/media/deepstation/df661f44-30ab-4e78-be4b-85a6014ac61d/deepstation/1j-ocr-20171221/"
train_data_dir = DATA_ROOT + 'train'
validation_data_dir = DATA_ROOT + 'val'

nb_train_samples = (80*11)
nb_val_samples = (20*11-6-1)
nb_epoch = 500

result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


if __name__ == '__main__':
    # VGG16モデルと学習済み重みをロード
    # Fully-connected層（FC）はいらないのでinclude_top=False）
    input_tensor = Input(shape=(img_rows, img_cols, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    # vgg16.summary()

    # FC層を構築
    # Flattenへの入力指定はバッチ数を除く
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))

    # 学習済みのFC層の重みをロード
    # top_model.load_weights(os.path.join(result_dir, 'bottleneck_fc_model.h5'))

    # VGG16とFCを接続
    model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

    # 最後のconv層の直前までの層をfreeze
    for layer in model.layers[:15]:
        layer.trainable = False

    # Fine-tuningのときはSGDの方がよい？
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    # Fine-tuning
    history = model.fit_generator(
        train_generator,
        steps_per_epoch = nb_train_samples,
        epochs =nb_epoch,
        validation_data = validation_generator,
        validation_steps = nb_val_samples)

    model.save_weights(os.path.join(result_dir, 'finetuning.h5'))
    save_history(history, os.path.join(result_dir, 'history_finetuning.txt'))
