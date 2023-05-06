'''
Created on 21-Apr-2023

@author: EZIGO
'''
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import train_config as config
from sklearn import metrics

valAug = ImageDataGenerator(rescale=1 / 255.0)

TestGen = valAug.flow_from_directory(config.TEST_PATH,
                                                target_size=(80, 80),
                                                batch_size=config.BS,
                                                class_mode='categorical',
                                                color_mode="rgb",
                                                shuffle=False
                                                )              
      
# json_file = open("FASNetSE.json", 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# model.load_weights("FASNetSE.hdf5")
# print("[INFO] evaluating network...")
# TestGen.reset()
# y_prob=model.predict(x=TestGen,steps=(5364//config.BS+1))
# y = np.argmax(y_prob, axis=1)
# print(metrics.classification_report(TestGen.classes, y,target_names=TestGen.class_indices.keys()))








json_file = open("MultiFTNetSE.json", 'r')
loaded_model_json = json_file.read()
json_file.close()

custom_objects = {'AdaptiveAveragePooling2D': tfa.layers.AdaptiveAveragePooling2D}
model = model_from_json(loaded_model_json, 
                        custom_objects = custom_objects,
                        )

model.load_weights("MultiFTNetSE.hdf5")
print("[INFO] evaluating network...")
TestGen.reset()
y_prob=model.predict(x=TestGen,steps=(5364//config.BS+1))[0]
y = np.argmax(y_prob, axis=1)
print(metrics.classification_report(TestGen.classes, y,target_names=TestGen.class_indices.keys()))