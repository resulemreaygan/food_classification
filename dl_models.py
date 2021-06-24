"""
Author: Resul Emre AYGAN
"""

import itertools
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from all_constants import AllConstant


class Operations:
    def __init__(self):
        self.all_const = AllConstant()
        self.input_shape = self.all_const.input_shape
        self.loaded_model = None

        if self.all_const.predict or self.all_const.evaluation:
            self.loaded_model = load_model(self.all_const.weight_path)

    def generate_data(self):
        train_data_generator = ImageDataGenerator(rescale=1. / 255,
                                                  rotation_range=40, width_shift_range=0.2,
                                                  height_shift_range=0.2, shear_range=0.2,
                                                  zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

        train_data_gen = train_data_generator.flow_from_directory(directory=self.all_const.train_set,
                                                                  target_size=self.input_shape,
                                                                  class_mode='categorical',
                                                                  shuffle=True,
                                                                  batch_size=self.all_const.batch_size)

        test_data_generator = ImageDataGenerator(rescale=1. / 255)

        test_data_gen = test_data_generator.flow_from_directory(directory=self.all_const.validation_set,
                                                                target_size=self.input_shape,
                                                                class_mode='categorical',
                                                                batch_size=self.all_const.batch_size)

        return train_data_gen, test_data_gen

    def plot_result(self, history):
        plt.plot(history.history["accuracy"])
        plt.plot(history.history['val_accuracy'])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
        date_now = str(datetime.now().strftime("%Y%m%d-%H%M%S"))

        fig_path = os.path.join(self.all_const.checkpoints_path + str(os.sep) + 'fig_' + date_now)
        plt.savefig(fig_path)
        # plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
        """
        given a sklearn confusion matrix (cm), make a nice plot

        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        target_names: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      see http://matplotlib.org/examples/color/colormaps_reference.html
                      plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                      If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                  # sklearn.metrics.confusion_matrix
                              normalize    = True,                # show proportions
                              target_names = y_labels_vals,       # list of names of the classes
                              title        = best_estimator_name) # title of graph

        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        """

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.savefig("confusion_matrix.png")
        # plt.show()

    def evaluation_data(self):
        evaluation_generator = ImageDataGenerator(rescale=1. / 255)

        evaluation_datagen = evaluation_generator.flow_from_directory(directory=self.all_const.evaluation_set,
                                                                      target_size=self.input_shape,
                                                                      color_mode="rgb", shuffle=False,
                                                                      class_mode='categorical', batch_size=1)

        y_pred = self.loaded_model.predict_generator(evaluation_datagen, steps=len(evaluation_datagen.filenames),
                                                     workers=1, use_multiprocessing=False)
        predicted_class = np.argmax(y_pred, axis=1)
        true_class = evaluation_datagen.classes
        class_labels = list(evaluation_datagen.class_indices.keys())

        cm = confusion_matrix(y_true=true_class, y_pred=predicted_class)

        self.plot_confusion_matrix(cm=cm, normalize=True, target_names=class_labels)

    @staticmethod
    def decode_predictions(preds, top_n=1):
        tags = ['bread', 'dairy_product', 'dessert', 'egg', 'fried_food',
                'meat', 'noodles_pasta', 'rice', 'seafood', 'soup', 'vegetable-fruit']

        assert len(preds.shape) == 2 and preds.shape[1] == 11
        pred_list = []

        for pred in preds:
            result = zip(tags, pred)
            result = sorted(result, key=lambda x: x[1], reverse=True)
            pred_list.append(result[:top_n])

        return pred_list

    def predict_image(self, image_path, model_path):
        if not os.path.exists(image_path):
            print("Image not found!")
            return 0

        loaded_image = load_img(image_path, target_size=self.all_const.input_shape)
        loaded_image = np.asarray(loaded_image)
        # plt.imshow(loaded_image)
        loaded_image = np.expand_dims(loaded_image, axis=0)

        if not os.path.exists(model_path):
            print("Model not found!")
            return 0

        if self.loaded_model is None:
            self.loaded_model = load_model(model_path)

        output = self.loaded_model.predict(loaded_image)

        decoded_output = self.decode_predictions(preds=output, top_n=3)

        return decoded_output


class VGG16:
    def __init__(self, input_shape):
        self.all_constant = AllConstant()
        self.model = Sequential()
        self.build_model(input_shape)

    def build_model(self, input_shape):
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                              input_shape=input_shape))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(units=4096, activation='relu'))
        self.model.add(Dense(units=4096, activation='relu'))
        self.model.add(Dense(units=11, activation='softmax'))

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.all_constant.lr),
                           loss=categorical_crossentropy, metrics=['accuracy'])

        self.model.summary()


class VGG19:
    def __init__(self, input_shape):
        self.all_constant = AllConstant()
        self.model = Sequential()
        self.build_model(input_shape)

    def build_model(self, input_shape):
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                              input_shape=input_shape))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(units=4096, activation='relu'))
        self.model.add(Dropout(0.25))  # 0.5
        self.model.add(Dense(units=4096, activation='relu'))
        self.model.add(Dropout(0.25))  # 0.5
        self.model.add(Dense(units=11, activation='softmax'))

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.all_constant.lr),
                           loss=categorical_crossentropy, metrics=['accuracy'])

        self.model.summary()


if __name__ == '__main__':
    a_c = AllConstant()
    op = Operations()

    if a_c.train:
        train_data, test_data = op.generate_data()

        if a_c.model_type == "VGG19":
            vgg19 = VGG19((224, 224, 3))
            model = vgg19.model

        elif a_c.model_type == "VGG16":
            model = VGG16((224, 224, 3))
        else:
            vgg19 = VGG19((224, 224, 3))
            model = vgg19.model

        temp_date = str(datetime.now().strftime("%Y%m%d-%H%M%S"))
        cp_name = a_c.model_type + "_" + temp_date + ".h5"

        cp_path = os.path.join(a_c.checkpoints_path, cp_name)

        checkpoint = ModelCheckpoint(cp_path, monitor='val_accuracy', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)

        early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

        hist = model.fit_generator(steps_per_epoch=2000 // a_c.batch_size, generator=train_data,
                                   validation_data=test_data, validation_steps=200 // a_c.batch_size,
                                   epochs=a_c.epochs, callbacks=[checkpoint, early])

        op.plot_result(history=hist)

    if a_c.evaluation:
        op.evaluation_data()

    if a_c.predict:
        results = op.predict_image(
            image_path=a_c.predict_path,
            model_path=a_c.weight_path)

        print(results)
