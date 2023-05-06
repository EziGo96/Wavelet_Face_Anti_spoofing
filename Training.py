'''
Created on 12-Apr-2023

@author: EZIGO
'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow_addons.optimizers import SGDW
from Models.WaveletFASNet import FASNetSE, MultiFTNetSE,MultiWTNetSE
from utils import FT, WT, jitter_preprocessing
import train_config as config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import save_model


totalTrain = len(list(paths.list_images(config.TRAIN_PATH)))
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))

trainAug = ImageDataGenerator(rescale=1 / 255.0,
                              rotation_range=90,
                              zoom_range=0.05,
                              width_shift_range=0.05,
                              height_shift_range=0.05,
                              brightness_range=(0.5, 1.5),
                              preprocessing_function=jitter_preprocessing,
                              # shear_range=0.05,
                              horizontal_flip=True,
                              fill_mode="nearest")

valAug = ImageDataGenerator(rescale=1 / 255.0)

# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,patience=3, min_lr=0.001)
def lr_schedule(epoch):
    maxEpochs=config.epochs
    baseLR=config.alpha
    factor=0.1
    if epoch<=5:
        alpha=baseLR
    # elif epoch>5 and epoch<=9:
    #     alpha=baseLR*factor
    # elif epoch>9 and epoch<=12:
    #     alpha=baseLR*(factor**4)
    else:
        alpha=baseLR*(factor)
    return alpha
reduce_lr=LearningRateScheduler(lr_schedule)  


# '''FASNetSE Training'''
# def train_generator(directory, batch_size, target_size, color_mode="rgb", shuffle=True, class_mode='categorical'):
#     # Create the base generator
#     base_generator = trainAug.flow_from_directory(directory,
#                                                   target_size=target_size,
#                                                   batch_size=batch_size,
#                                                   class_mode=class_mode,
#                                                   color_mode=color_mode,
#                                                   shuffle=shuffle
#                                                   )
#
#     while True:
#         # Get the next batch of images and labels
#         x_batch, y_batch_class = next(base_generator)
#         # Calculate the custom target variables
#         yield x_batch, y_batch_class
#
# def val_generator(directory, batch_size, target_size, color_mode="rgb", shuffle=False, class_mode='categorical'):
#     # Create the base generator
#     base_generator = valAug.flow_from_directory(directory,
#                                                 target_size=target_size,
#                                                 batch_size=batch_size,
#                                                 class_mode=class_mode,
#                                                 color_mode=color_mode,
#                                                 shuffle=shuffle
#                                                 )
#
#     while True:
#         # Get the next batch of images and labels
#         x_batch, y_batch_class = next(base_generator)
#         # Calculate the custom target variables
#         yield x_batch, y_batch_class       
#
# # initialize the training generator
# trainGen = train_generator(config.TRAIN_PATH,
#                             batch_size=config.BS,
#                             target_size=(80, 80))
# # initialize the validation generator
# valGen = val_generator(config.VAL_PATH,
#                         batch_size=config.BS,
#                         target_size=(80, 80))
#
#
#
#
# mcp_save = ModelCheckpoint("FASNetSE.hdf5", save_best_only=True, monitor='val_loss', mode='min')
# callbacks = [mcp_save,reduce_lr]
# opt=SGDW(learning_rate=config.alpha, weight_decay=5e-4,momentum=0.9)
#
# model = FASNetSE.build((80,80,3), classes=3, drop_p=0.75)
# # model.summary()    
# # plot_model(model,to_file="FASNetSE.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True) 
#
# model.compile(optimizer=opt,
#               loss=['categorical_crossentropy'],
#               metrics=['accuracy'])
#
# H = model.fit(trainGen,
#               validation_data=valGen,
#               epochs=config.epochs,
#               steps_per_epoch=19310//config.BS,
#               validation_steps=2145//config.BS,
#               callbacks=callbacks)
# df=pd.DataFrame()
# df.from_dict(H.history).to_csv("TrainingFASNetSE.csv",index=False)
# model_json = model.to_json(indent=3)
# with open("FASNetSE.json", "w") as json_file:
#     json_file.write(model_json)
# save_model(model, "FASNetSE.hp5", save_format="h5")






# '''MultiFTNetSE Training'''
# def train_generator(directory, batch_size, target_size, color_mode="rgb", shuffle=True, class_mode='categorical'):
#     # Create the base generator
#     base_generator = trainAug.flow_from_directory(directory,
#                                                   target_size=target_size,
#                                                   batch_size=batch_size,
#                                                   class_mode=class_mode,
#                                                   color_mode=color_mode,
#                                                   shuffle=shuffle
#                                                   )
#
#     while True:
#         # Get the next batch of images and labels
#         x_batch, y_batch_class = next(base_generator)
#         # Calculate the custom target variables
#         y_batch_FT = FT(np.array(x_batch)).predict(x_batch)
#         yield x_batch, [y_batch_class, y_batch_FT]
#
# def val_generator(directory, batch_size, target_size, color_mode="rgb", shuffle=False, class_mode='categorical'):
#     # Create the base generator
#     base_generator = valAug.flow_from_directory(directory,
#                                                 target_size=target_size,
#                                                 batch_size=batch_size,
#                                                 class_mode=class_mode,
#                                                 color_mode=color_mode,
#                                                 shuffle=shuffle
#                                                 )
#
#     while True:
#         # Get the next batch of images and labels
#         x_batch, y_batch_class = next(base_generator)
#         # Calculate the custom target variables
#         y_batch_FT = FT(np.array(x_batch)).predict(x_batch)
#         yield x_batch, [y_batch_class, y_batch_FT]        
#
# # initialize the training generator
# trainGen = train_generator(config.TRAIN_PATH,
#                             batch_size=config.BS,
#                             target_size=(80, 80))
# # initialize the validation generator
# valGen = val_generator(config.VAL_PATH,
#                         batch_size=config.BS,
#                         target_size=(80, 80))
#
# # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=5, min_lr=0.001)
# mcp_save = ModelCheckpoint("MultiFTNetSE.hdf5", save_best_only=True, monitor='val_Softmax_loss', mode='min')
# callbacks = [mcp_save,reduce_lr]
# opt=SGDW(learning_rate=config.alpha, weight_decay=5e-4,momentum=0.9)
#
# model = MultiFTNetSE.build((80,80,3), classes=3, drop_p=0.75, training=True)
# # model.summary()    
# # plot_model(model,to_file="MultiFTNetSE.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True) 
#
# model.compile(optimizer=opt,
#               loss={'Softmax': 'categorical_crossentropy','FTGenerator': 'mse'},
#               loss_weights={'Softmax': 0.5, 'FTGenerator': 0.5},
#               metrics={'Softmax': 'accuracy'})
#
# H = model.fit(trainGen,
#               validation_data=valGen,
#               epochs=config.epochs,
#               steps_per_epoch=19310//config.BS,
#               validation_steps=2145//config.BS,
#               callbacks=callbacks)
# df=pd.DataFrame()
# df.from_dict(H.history).to_csv("TrainingMultiFTNetSE.csv",index=False)
# model_json = model.to_json(indent=3)
# with open("MultiFTNetSE.json", "w") as json_file:
#     json_file.write(model_json)
# save_model(model, "MultiFTNetSE.hp5", save_format="h5")






'''MultiWTNetSE training'''

def train_generator(directory, batch_size, target_size, color_mode="rgb", shuffle=True, class_mode='categorical'):
    # Create the base generator
    base_generator = trainAug.flow_from_directory(directory,
                                                  target_size=target_size,
                                                  batch_size=batch_size,
                                                  class_mode=class_mode,
                                                  color_mode=color_mode,
                                                  shuffle=shuffle
                                                  )

    while True:
        # Get the next batch of images and labels
        x_batch, y_batch_class = next(base_generator)
        # Calculate the custom target variables
        y_batch_WT = WT(np.array(x_batch)).predict(x_batch)
        yield x_batch, [y_batch_class, y_batch_WT]

def val_generator(directory, batch_size, target_size, color_mode="rgb", shuffle=False, class_mode='categorical'):
    # Create the base generator
    base_generator = valAug.flow_from_directory(directory,
                                                target_size=target_size,
                                                batch_size=batch_size,
                                                class_mode=class_mode,
                                                color_mode=color_mode,
                                                shuffle=shuffle
                                                )

    while True:
        # Get the next batch of images and labels
        x_batch, y_batch_class = next(base_generator)
        # Calculate the custom target variables
        y_batch_WT = WT(np.array(x_batch)).predict(x_batch)
        yield x_batch, [y_batch_class, y_batch_WT]        

# initialize the training generator
trainGen = train_generator(config.TRAIN_PATH,
                            batch_size=config.BS,
                            target_size=(80, 80))
# initialize the validation generator
valGen = val_generator(config.VAL_PATH,
                        batch_size=config.BS,
                        target_size=(80, 80))

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=5, min_lr=0.0001)
mcp_save = ModelCheckpoint("MultiWTNetSE.hdf5", save_best_only=True, monitor='Softmax_loss', mode='min')
callbacks = [mcp_save,reduce_lr]
opt=SGDW(learning_rate=config.alpha, weight_decay=5e-4,momentum=0.9)

model = MultiWTNetSE.build((80,80,3), classes=3, drop_p=0.75, training=True)
# model.summary()    
# plot_model(model,to_file="MultiWTNetSE.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True) 

model.compile(optimizer=opt,
              loss={'Softmax': 'categorical_crossentropy','WTGenerator': 'mse'},
              loss_weights={'Softmax': 0.5, 'WTGenerator': 0.5},
              metrics={'Softmax': 'accuracy'})

H = model.fit(trainGen,
              validation_data=valGen,
              epochs=config.epochs,
              steps_per_epoch=19310//config.BS,
              validation_steps=2145//config.BS,
              callbacks=callbacks)
df=pd.DataFrame()
df.from_dict(H.history).to_csv("TrainingMultiWTNetSE.csv",index=False)
model_json = model.to_json(indent=3)
with open("MultiWTNetSE.json", "w") as json_file:
    json_file.write(model_json)
save_model(model, "MultiWTNetSE.hp5", save_format="h5")
