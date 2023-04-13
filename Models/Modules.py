'''
Created on 11-Apr-2023

@author: EZIGO
'''
from tensorflow_addons.layers import AdaptiveAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import ReLU
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from Models.Config import keep_dict

'''global parameters needed to test the modules'''
chanDim=-1
keep=keep_dict['1.8M_']

class Conv_block:
    @staticmethod
    def build(inputs, filters, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), groups=1, chanDim=-1, name="Conv_block"):
        inputShape=inputs.shape[1:]
        inputs=Input(shape=inputShape,name='inputs')
        x=ZeroPadding2D(padding=padding)(inputs)
        x=Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='VALID', use_bias=False, groups=groups)(x)
        x=BatchNormalization(axis=chanDim)(x)
        x=PReLU()(x)
        model=Model(inputs=inputs, outputs=x, name=name)
        return model
    
# inputs=Input((80,80,3))    
# model = Conv_block.build(inputs,filters=keep[0], kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), chanDim=chanDim)
# model.summary()    
# plot_model(model,to_file="Conv_block.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True)

class Linear_conv_block:
    @staticmethod    
    def build(inputs, filters, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), groups=1, chanDim=-1, name="Linear_conv_block"):
        inputShape=inputs.shape[1:]
        inputs=Input(shape=inputShape,name='inputs')
        x=Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='VALID', use_bias=False, groups=groups)(inputs)
        x=BatchNormalization(axis=chanDim)(x)
        model=Model(inputs=inputs, outputs=x, name=name)
        return model
       
# inputs=Input((5,5,512))     
# model = Linear_conv_block.build(inputs, filters=keep[48], kernel_size=(5,5), strides=(1, 1), padding=(0, 0), groups=keep[48], chanDim=chanDim)
# model.summary()    
# plot_model(model,to_file="Linear_conv_block.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True)   

class DepthWise:    
    @staticmethod
    def build(inputs, c1, c2, c3, residual=False, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), groups=1, chanDim=-1, name="DepthWise"):
        inputShape=inputs.shape[1:]
        inputs=Input(shape=inputShape,name='inputs')
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        if residual:
            shortcut=inputs
        x=Conv_block.build(inputs, filters=c1_out, kernel_size=(1, 1), padding=(0, 0), strides=(1, 1), chanDim=chanDim, name="conv1" )(inputs)
        x=Conv_block.build(x, filters=c2_out, kernel_size=kernel_size, padding=padding, strides=strides , groups=c2_in, chanDim=chanDim, name="conv2" )(x)
        x=Linear_conv_block.build(x, filters=c3_out, kernel_size=(1, 1), padding=(0, 0), strides=(1, 1), chanDim=chanDim)(x)
        if residual:
            x+=shortcut
        model=Model(inputs=inputs, outputs=x, name=name)
        return model
    
'''local  parameters needed to test the DepthWise module'''    
c1 = [(keep[1], keep[2])]
c2 = [(keep[2], keep[3])]
c3 = [(keep[3], keep[4])]
    
# inputs=Input((40,40,32))  
# model = DepthWise.build(inputs,c1[0], c2[0], c3[0], kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), groups=keep[40], chanDim=chanDim)
# model.summary()    
# plot_model(model,to_file="DepthWise.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True) 

'''local  parameters needed to test the SE,DepthWiseSE, ResidualSE module''' 
c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])] 
c1_tuple = c1[0]
c2_tuple = c2[0]
c3_tuple = c3[0]
c3_out=c3_tuple[1]
se_reduct=4

class SEModule:
    @staticmethod
    def build(inputs, channels, reduction, chanDim=-1, name="SEModule"):
        inputShape=inputs.shape[1:]
        inputs=Input(shape=inputShape,name='inputs')
        block_input=inputs
        if chanDim==-1:
            data_format="channels_last"
        else:
            data_format="channels_first"
        x=AdaptiveAveragePooling2D((1,1),data_format=data_format)(inputs)
        x=Conv2D(filters=channels // reduction, kernel_size=(1,1), padding='VALID', use_bias=False)(x)
        x=BatchNormalization(axis=chanDim)(x)
        x=ReLU()(x)
        x=Conv2D(filters=channels, kernel_size=(1,1), padding='VALID', use_bias=False)(x)
        x=BatchNormalization(axis=chanDim)(x)
        x=sigmoid(x)
        x*=block_input
        model=Model(inputs=inputs, outputs=x, name=name)
        return model
    
# inputs=Input((20,20,64))  
# model = SEModule.build(inputs, c3_out, reduction=se_reduct, chanDim=chanDim)
# model.summary()    
# plot_model(model,to_file="SEModule.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True) 

   
class DepthWiseSE:
    @staticmethod
    def build(inputs, c1, c2, c3, residual=False, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), groups=1, se_reduct=8, chanDim=-1, name="DepthWiseSE"):
        inputShape=inputs.shape[1:]
        inputs=Input(shape=inputShape,name='inputs')
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        if residual:
            shortcut=inputs
        x=Conv_block.build(inputs, filters=c1_out, kernel_size=(1, 1), padding=(0, 0), strides=(1, 1), chanDim=chanDim, name="conv1")(inputs)
        x=Conv_block.build(x, filters=c2_out, kernel_size=kernel_size, padding=padding, strides=strides , groups=c2_in, chanDim=chanDim,name="conv2")(x)
        x=Linear_conv_block.build(x, filters=c3_out, kernel_size=(1, 1), padding=(0, 0), strides=(1, 1), chanDim=chanDim)(x)
        if residual:
            x=SEModule.build(x, c3_out, se_reduct,chanDim=chanDim)(x)
            x+=shortcut
        model=Model(inputs=inputs, outputs=x, name=name)
        return model
    
# inputs=Input((20,20,64))   
# model = DepthWiseSE.build(inputs, c1_tuple, c2_tuple, c3_tuple, residual=True, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), groups=1, se_reduct=4, chanDim=chanDim)
# model.summary()    
# plot_model(model,to_file="DepthWiseSE.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True) 
   
class ResidualSE:    
    @staticmethod
    def build(inputs, c1, c2, c3, num_block, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), groups=1, se_reduct=4, chanDim=-1, name="ResidualSE"):
        inputShape=inputs.shape[1:]
        inputs=Input(shape=inputShape,name='inputs')
        for i in range(num_block-1):
            if i==0:
                x=inputs
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            x=DepthWiseSE.build(x, c1_tuple, c2_tuple, c3_tuple, residual=True, kernel_size=kernel_size, strides=strides, padding=padding, groups=groups, se_reduct=se_reduct, chanDim=chanDim, name="DepthWiseSE_"+str(i+1))(x)
        c1_tuple = c1[num_block-1]
        c2_tuple = c2[num_block-1]
        c3_tuple = c3[num_block-1]
        x=DepthWise.build(x, c1_tuple, c2_tuple, c3_tuple, residual=True, kernel_size=kernel_size, strides=strides, padding=padding, groups=groups, chanDim=chanDim)(x)
        model=Model(inputs=inputs, outputs=x, name=name)
        return model
    
# inputs=Input((20,20,64)) 
# model=ResidualSE.build(inputs,c1, c2, c3, num_block=4, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), groups=keep[4], chanDim=chanDim)
# model.summary()    
# plot_model(model,to_file="ResidualSE.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True) 

class FTGenerator:
    @staticmethod
    def build(inputs,out_channels=1,chanDim=-1,name="FTGenerator"):
        inputShape=inputs.shape[1:]
        inputs=Input(shape=inputShape,name='inputs')
        x=ZeroPadding2D(padding=(1,1))(inputs)
        x=Conv2D(128, kernel_size=(3, 3), padding='VALID')(x)
        x=BatchNormalization(axis=chanDim)(x)
        x=ReLU()(x)
    
        x=ZeroPadding2D(padding=(1,1))(x)
        x=Conv2D(64, kernel_size=(3, 3), padding='VALID')(x)
        x=BatchNormalization(axis=chanDim)(x)
        x=ReLU()(x)

        x=ZeroPadding2D(padding=(1,1))(x)
        x=Conv2D(out_channels, kernel_size=(3, 3), padding='VALID')(x)
        x=BatchNormalization(axis=chanDim)(x)
        x=ReLU()(x)
        
        model=Model(inputs=inputs, outputs=x, name=name)
        return model
        
# inputs=Input((10,10,128)) 
# model=FTGenerator.build(inputs,chanDim=chanDim)
# model.summary()    
# plot_model(model,to_file="FTGenerator.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True)

class WTGenerator:
    @staticmethod
    def build(inputs,out_channels=4,chanDim=-1,name="WTGenerator"):
        inputShape=inputs.shape[1:]
        inputs=Input(shape=inputShape,name='inputs')
        
        x=ZeroPadding2D(padding=(1,1))(inputs)
        x=Conv2D(128, kernel_size=(3, 3), padding='VALID')(x)
        x=BatchNormalization(axis=chanDim)(x)
        x=ReLU()(x)
        
        x=ZeroPadding2D(padding=(1,1))(x)
        x=Conv2D(64, kernel_size=(3, 3), padding='VALID')(x)
        x=BatchNormalization(axis=chanDim)(x)
        x=ReLU()(x)
        
        x=ZeroPadding2D(padding=(1,1))(x)
        x=Conv2D(out_channels, kernel_size=(3, 3), padding='VALID')(x)
        x=BatchNormalization(axis=chanDim)(x)
        x=ReLU()(x)
        
        model=Model(inputs=inputs, outputs=x, name=name)
        return model 
    
# inputs=Input((10,10,128)) 
# model=WTGenerator.build(inputs,chanDim=chanDim)
# model.summary()    
# plot_model(model,to_file="WTGenerator.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True)
