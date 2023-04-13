'''
Created on 26-Mar-2023

@author: EZIGO
'''



from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from Models.Modules import Conv_block,Linear_conv_block,DepthWise,ResidualSE,FTGenerator,WTGenerator
from Models.Config import keep_dict


'''Model 1: without any auxillary pathway'''
class FASNetSE:
    @staticmethod
    def build(input_dim, classes, keep=keep_dict['1.8M_'], embedding_size=128, conv6_kernel=(5, 5), drop_p=0.0, reg=0.002):
        (width, height, channels)=input_dim
        inputShape = (width, height, channels)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (channels, height, width)
            chanDim = 1
        
        '''Input_layer'''
        inputs=Input(shape=inputShape,name='inputs')
        
        '''conv1'''
        x=Conv_block.build(inputs,filters=keep[0], kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), chanDim=chanDim, name="conv1")(inputs)
        
        '''conv2'''
        x=Conv_block.build(x,filters=keep[1], kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), groups=keep[1], chanDim=chanDim, name="conv2")(x)
        
        '''DWconv23'''
        c1 = [(keep[1], keep[2])]
        c2 = [(keep[2], keep[3])]
        c3 = [(keep[3], keep[4])]
        x=DepthWise.build(x, c1[0], c2[0], c3[0], kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), groups=keep[3], chanDim=chanDim, name="DWconv23")(x)
        
        '''ResidualSE3'''
        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]
        x=ResidualSE.build(x,c1, c2, c3, num_block=4, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), groups=keep[4], chanDim=chanDim, name="ResidualSE3")(x)
        
        '''DWconv34'''
        c1 = [(keep[16], keep[17])]
        c2 = [(keep[17], keep[18])]
        c3 = [(keep[18], keep[19])]
        x=DepthWise.build(x, c1[0], c2[0], c3[0], kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), groups=keep[19], chanDim=chanDim, name="DWconv34")(x)
        
        '''ResidualSE4'''
        c1 = [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]), 
              (keep[28], keep[29]),(keep[31], keep[32]), (keep[34], keep[35])]
        c2 = [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]), 
              (keep[29], keep[30]), (keep[32], keep[33]), (keep[35], keep[36])]
        c3 = [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]), 
              (keep[30], keep[31]), (keep[33], keep[34]), (keep[36], keep[37])]
        x=ResidualSE.build(x, c1, c2, c3, num_block=6, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), groups=keep[19], chanDim=chanDim, name="ResidualSE4")(x)
        
        '''DWconv45'''
        c1 = [(keep[37], keep[38])]
        c2 = [(keep[38], keep[39])]
        c3 = [(keep[39], keep[40])]
        x=DepthWise.build(x,c1[0], c2[0], c3[0], kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), groups=keep[40], chanDim=chanDim, name="DWconv45")(x)
        
        '''ResidualSE5'''
        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        x=ResidualSE.build(x,c1, c2, c3, num_block=2, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), groups=keep[40], chanDim=chanDim, name="ResidualSE5")(x)
        
        '''conv6'''
        x=Conv_block.build(x, filters=keep[47], kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), chanDim=chanDim, name="conv6")(x)
        
        '''conv6_proj'''
        x=Linear_conv_block.build(x, filters=keep[48], kernel_size=conv6_kernel, strides=(1, 1), padding=(0, 0), groups=keep[48], chanDim=chanDim, name="conv6_proj")(x)
        
        '''FC_layer7'''
        x=Flatten()(x)
        x=Dense(embedding_size, kernel_regularizer=l2(reg), use_bias=False, name="FC_layer7")(x)
        x=BatchNormalization(axis=chanDim)(x)
        x=Dropout(drop_p)(x)
        
        '''FC_layer_8'''
        x=Dense(classes, kernel_regularizer=l2(reg), use_bias=False, name="FC_layer_8")(x)
        
        '''SoftMax layer'''
        x = Activation("softmax")(x)
        
        model=Model(inputs=inputs, outputs=x, name="FASNet")
        return model

        
# model = FASNetSE.build((80,80,3), classes=3, drop_p=0.75)
# model.summary()    
# plot_model(model,to_file="FASNetSE.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True)    

'''Model 2: with FT auxillary pathway'''
class MultiFTNetSE:
    @staticmethod
    def build(input_dim, classes, keep=keep_dict['1.8M_'], embedding_size=128, conv6_kernel=(5, 5), drop_p=0.0, reg=0.002, training=True):
        (width, height, channels)=input_dim
        inputShape = (width, height, channels)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (channels, height, width)
            chanDim = 1
        
        '''Input_layer'''
        inputs=Input(shape=inputShape,name='inputs')
        
        '''conv1'''
        x=Conv_block.build(inputs,filters=keep[0], kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), chanDim=chanDim, name="conv1")(inputs)
        
        '''conv2'''
        x=Conv_block.build(x,filters=keep[1], kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), groups=keep[1], chanDim=chanDim, name="conv2")(x)
        
        '''DWconv23'''
        c1 = [(keep[1], keep[2])]
        c2 = [(keep[2], keep[3])]
        c3 = [(keep[3], keep[4])]
        x=DepthWise.build(x, c1[0], c2[0], c3[0], kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), groups=keep[3], chanDim=chanDim, name="DWconv23")(x)
        
        '''ResidualSE3'''
        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]
        x=ResidualSE.build(x,c1, c2, c3, num_block=4, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), groups=keep[4], chanDim=chanDim, name="ResidualSE3")(x)
        
        '''DWconv34'''
        c1 = [(keep[16], keep[17])]
        c2 = [(keep[17], keep[18])]
        c3 = [(keep[18], keep[19])]
        x=DepthWise.build(x, c1[0], c2[0], c3[0], kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), groups=keep[19], chanDim=chanDim, name="DWconv34")(x)
        
        '''ResidualSE4'''
        c1 = [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]), 
              (keep[28], keep[29]),(keep[31], keep[32]), (keep[34], keep[35])]
        c2 = [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]), 
              (keep[29], keep[30]), (keep[32], keep[33]), (keep[35], keep[36])]
        c3 = [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]), 
              (keep[30], keep[31]), (keep[33], keep[34]), (keep[36], keep[37])]
        x=ResidualSE.build(x, c1, c2, c3, num_block=6, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), groups=keep[19], chanDim=chanDim, name="ResidualSE4")(x)
        
        '''FT Auxillary path'''
        if training:
            FT_path=FTGenerator.build(x,out_channels=1, chanDim=chanDim)
            x_ft=FT_path(x)
            
        '''DWconv45'''
        c1 = [(keep[37], keep[38])]
        c2 = [(keep[38], keep[39])]
        c3 = [(keep[39], keep[40])]
        x=DepthWise.build(x,c1[0], c2[0], c3[0], kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), groups=keep[40], chanDim=chanDim, name="DWconv45")(x)
        
        '''ResidualSE5'''
        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        x=ResidualSE.build(x,c1, c2, c3, num_block=2, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), groups=keep[40], chanDim=chanDim, name="ResidualSE5")(x)
        
        '''conv6'''
        x=Conv_block.build(x, filters=keep[47], kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), chanDim=chanDim, name="conv6")(x)
        
        '''conv6_proj'''
        x=Linear_conv_block.build(x, filters=keep[48], kernel_size=conv6_kernel, strides=(1, 1), padding=(0, 0), groups=keep[48], chanDim=chanDim, name="conv6_proj")(x)
        
        '''FC_layer7'''
        x=Flatten()(x)
        x=Dense(embedding_size, kernel_regularizer=l2(reg), use_bias=False, name="FC_layer7")(x)
        x=BatchNormalization(axis=chanDim)(x)
        x=Dropout(drop_p)(x)
        
        '''FC_layer_8'''
        x=Dense(classes, kernel_regularizer=l2(reg), use_bias=False, name="FC_layer_8")(x)
        
        '''SoftMax layer'''
        x = Activation("softmax", name="Softmax")(x)
        
        if training:
            model=Model(inputs=inputs, outputs=[x, x_ft], name="MultiFTNetSE")
        else:
            model=Model(inputs=inputs, outputs=x, name="MultiFTNetSE")
        return model

        
# model = MultiFTNetSE.build((80,80,3), classes=3, drop_p=0.75, training=True)
# model.summary()    
# plot_model(model,to_file="MultiFTNetSE.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True)

'''Model 2: with WT auxillary pathway'''
class MultiWTNetSE:
    @staticmethod
    def build(input_dim, classes, keep=keep_dict['1.8M_'], embedding_size=128, conv6_kernel=(5, 5), drop_p=0.0, reg=0.002, training=True):
        (width, height, channels)=input_dim
        inputShape = (width, height, channels)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (channels, height, width)
            chanDim = 1
        
        '''Input_layer'''
        inputs=Input(shape=inputShape,name='inputs')
        
        '''conv1'''
        x=Conv_block.build(inputs,filters=keep[0], kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), chanDim=chanDim, name="conv1")(inputs)
        
        '''conv2'''
        x=Conv_block.build(x,filters=keep[1], kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), groups=keep[1], chanDim=chanDim, name="conv2")(x)
        
        '''DWconv23'''
        c1 = [(keep[1], keep[2])]
        c2 = [(keep[2], keep[3])]
        c3 = [(keep[3], keep[4])]
        x=DepthWise.build(x, c1[0], c2[0], c3[0], kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), groups=keep[3], chanDim=chanDim, name="DWconv23")(x)
        
        '''ResidualSE3'''
        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]
        x=ResidualSE.build(x,c1, c2, c3, num_block=4, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), groups=keep[4], chanDim=chanDim, name="ResidualSE3")(x)
        
        '''DWconv34'''
        c1 = [(keep[16], keep[17])]
        c2 = [(keep[17], keep[18])]
        c3 = [(keep[18], keep[19])]
        x=DepthWise.build(x, c1[0], c2[0], c3[0], kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), groups=keep[19], chanDim=chanDim, name="DWconv34")(x)
        
        '''ResidualSE4'''
        c1 = [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]), 
              (keep[28], keep[29]),(keep[31], keep[32]), (keep[34], keep[35])]
        c2 = [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]), 
              (keep[29], keep[30]), (keep[32], keep[33]), (keep[35], keep[36])]
        c3 = [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]), 
              (keep[30], keep[31]), (keep[33], keep[34]), (keep[36], keep[37])]
        x=ResidualSE.build(x, c1, c2, c3, num_block=6, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), groups=keep[19], chanDim=chanDim, name="ResidualSE4")(x)
        
        '''WT Auxillary path'''
        if training:
            WTpath=WTGenerator.build(x,out_channels=4, chanDim=chanDim)
            x_wt=WTpath(x)
        
        '''DWconv45'''
        c1 = [(keep[37], keep[38])]
        c2 = [(keep[38], keep[39])]
        c3 = [(keep[39], keep[40])]
        x=DepthWise.build(x,c1[0], c2[0], c3[0], kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), groups=keep[40], chanDim=chanDim, name="DWconv45")(x)
        
        '''ResidualSE5'''
        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        x=ResidualSE.build(x,c1, c2, c3, num_block=2, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), groups=keep[40], chanDim=chanDim, name="ResidualSE5")(x)
        
        '''conv6'''
        x=Conv_block.build(x, filters=keep[47], kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), chanDim=chanDim, name="conv6")(x)
        
        '''conv6_proj'''
        x=Linear_conv_block.build(x, filters=keep[48], kernel_size=conv6_kernel, strides=(1, 1), padding=(0, 0), groups=keep[48], chanDim=chanDim, name="conv6_proj")(x)
        
        '''FC_layer7'''
        x=Flatten()(x)
        x=Dense(embedding_size, kernel_regularizer=l2(reg), use_bias=False, name="FC_layer7")(x)
        x=BatchNormalization(axis=chanDim)(x)
        x=Dropout(drop_p)(x)
        
        '''FC_layer_8'''
        x=Dense(classes, kernel_regularizer=l2(reg), use_bias=False, name="FC_layer_8")(x)
        
        '''SoftMax layer'''
        x = Activation("softmax", name="Softmax")(x)
        
        if training:
            model=Model(inputs=inputs, outputs=[x, x_wt], name="MultiWTNetSE")
        else:
            model=Model(inputs=inputs, outputs=x, name="MultiWTNetSE")
        return model

        
# model = MultiWTNetSE.build((80,80,3), classes=3, drop_p=0.75, training=True)
# model.summary()    
# plot_model(model,to_file="MultiWTNetSE.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True) 

