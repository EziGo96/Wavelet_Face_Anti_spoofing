tensorflow-wavelets is an implementation of Custom Layers for Neural Networks:
- *Discrete Wavelets Transform Layer*
- *Duel Tree Complex Wavelets Transform Layer*
- *Multi Wavelets Transform Layer*



# 
```
git clone https://github.com/Timorleiderman/tensorflow-wavelets.git
cd tensorflow-wavelets
git submodule update --init --recursive
```
## Installation

```
pip install tensorflow-wavelets
```
# Usage
```
import tensorflow_wavelets.Layers.DWT as DWT
import tensorflow_wavelets.Layers.DTCWT as DTCWT
import tensorflow_wavelets.Layers.DMWT as DMWT

# Custom Activation function Layer
import tensorflow_wavelets.Layers.Threshold as Threshold
```

# Examples
## DWT(name="haar", concat=0)
### "name" can be found in pywt.wavelist(family)
### concat = 0 means to split to 4 smaller layers

```
from tensorflow import keras
model = keras.Sequential()
model.add(keras.Input(shape=(28, 28, 1)))
model.add(DWT.DWT(name="haar",concat=0))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(nb_classes, activation="softmax"))
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    dwt_9_haar (DWT)             (None, 14, 14, 4)         0
    _________________________________________________________________
    flatten_9 (Flatten)          (None, 784)               0
    _________________________________________________________________
    dense_9 (Dense)              (None, 10)                7850
    =================================================================
    Total params: 7,850
    Trainable params: 7,850
    Non-trainable params: 0
    _________________________________________________________________

### name = "db4" concat = 1
```

model = keras.Sequential()
model.add(layers.InputLayer(input_shape=(28, 28, 1)))
model.add(DWT(name="db4", concat=1))
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    dwt_db4 (DWT)                (None, 34, 34, 1)         0
    =================================================================
    Total params: 0
    Trainable params: 0
    Non-trainable params: 0
    _________________________________________________________________

# DMWT
### functional example with Sure Threshold
```
from tensorflow.keras import layers
from tensorflow.keras.models import Model

x_inp = layers.Input(shape=(512, 512, 1))
x = DMWT("ghm")(x_inp)
x = Threshold.Threshold(algo='sure', mode='hard')(x) # use "soft" or "hard"
x = IDMWT("ghm")(x)
model = Model(x_inp, x, name="MyModel")
model.summary()
```
    Model: "MyModel"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         [(None, 512, 512, 1)]     0
    _________________________________________________________________
    dmwt (DMWT)                  (None, 1024, 1024, 1)     0
    _________________________________________________________________
    sure_threshold (SureThreshol (None, 1024, 1024, 1)     0
    _________________________________________________________________
    idmwt (IDMWT)                (None, 512, 512, 1)       0
    =================================================================
    Total params: 0
    Trainable params: 0
    Non-trainable params: 0
    _________________________________________________________________

**Free Software, Hell Yeah!**
