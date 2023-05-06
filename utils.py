'''
Created on 12-Apr-2023

@author: EZIGO
'''
import tensorflow as tf
from tensorflow.signal import fft2d
from tensorflow.keras.layers import BatchNormalization 
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow_wavelets.Layers import DWT



'''GPU FT implemetation'''
def FT(inputs):
    inputShape=inputs.shape[1:]
    inputs=Input(shape=inputShape,name='inputs')
    x=tf.image.rgb_to_grayscale(inputs)
    x=tf.cast(x,tf.complex64)
    x=fft2d(x)
    x=tf.signal.fftshift(x)
    x=tf.math.log(tf.math.abs(x)+1)
    min_x=tf.reduce_min(x)
    max_x=tf.reduce_max(x)
    x=(x - min_x) / (max_x - min_x)
    x=tf.image.resize_with_pad(x,10,10,method=tf.image.ResizeMethod.BILINEAR,antialias=False)
    
    model=Model(inputs=inputs,outputs=x,name="FT")
    return model

'''test code for GPU FT'''
# inputs=Input((80,80,3))    
# model = FT(inputs)
# model.summary()    
# plot_model(model,to_file="FT.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True)

# img_path="./datasets/rgb_80x80/1/fake_1_17.jpg"
# x=cv2.imread(img_path)
# x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
#
# plt.figure()
# plt.imshow(x)
#
# x_ft=FT(np.array([x])).predict(np.array([x]))
#
# plt.figure()
# plt.imshow(x_ft[0],cmap = 'gray')

'''GPU WT implemetation'''
def WT(inputs,name="db14",concat=1):
    inputShape=inputs.shape[1:]
    inputs=Input(shape=inputShape,name='inputs')
    x=tf.image.rgb_to_grayscale(inputs)
    # x=tf.cast(x,tf.complex64)
    x=DWT.DWT(name=name,concat=concat)(x)
    min_x=tf.reduce_min(x)
    max_x=tf.reduce_max(x)
    x=(x - min_x) / (max_x - min_x)
    x=tf.image.resize_with_pad(x,80,80,method=tf.image.ResizeMethod.BILINEAR,antialias=False)
    
    model=Model(inputs=inputs,outputs=x,name="WT")
    return model

# '''test code for GPU WT'''
# inputs=Input((80,80,3))    
# model = WT(inputs,concat=1)
# model.summary()    
# plot_model(model,to_file="WT.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True)
#
#
# img_path="./datasets/rgb_80x80/1/fake_1_17.jpg"
# x=cv2.imread(img_path)
# x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
#
# plt.figure()
# plt.imshow(x)
#
# x_wt=WT(np.array([x])).predict(np.array([x]))
# print(x_wt.shape)
# plt.figure()
# plt.imshow(x_wt[0],cmap = 'gray')
#
# plt.show()

def contrast_jitter(img, min_contrast=0.5, max_contrast=1.5):
    contrast = np.random.uniform(min_contrast, max_contrast)
    img = img.astype(np.float32)
    img = img * contrast
    img = np.clip(img, 0, 255)
    return img

def jitter_preprocessing(img):
    img = contrast_jitter(img)
    return img
