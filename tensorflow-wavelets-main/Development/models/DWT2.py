import os
import cv2
import math
import pywt
import numpy as np
from utils import mse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist, cifar10

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # for tensor flow warning
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class DWT(layers.Layer):
    def __init__(self, name='haar', **kwargs):
        super(DWT, self).__init__(**kwargs)
        self._name = self.name + "_" + name
        # get filter coeffs from 3rd party lib
        wavelet = pywt.Wavelet(name)
        self.dec_len = wavelet.dec_len

        # decomposition filter low pass and hight pass coeffs
        db2_lpf = wavelet.dec_lo
        db2_hpf = wavelet.dec_hi

        # covert filters into tensors and reshape for convolution math
        db2_lpf = tf.constant(db2_lpf[::-1])
        self.db2_lpf = tf.reshape(db2_lpf, (1, wavelet.dec_len, 1, 1))

        db2_hpf = tf.constant(db2_hpf[::-1])
        self.db2_hpf = tf.reshape(db2_hpf, (1, wavelet.dec_len, 1, 1))

        self.conv_type = "VALID"
        self.border_padd = "SYMMETRIC"

    def build(self, input_shape):
        # filter dims should be bigger if input is not gray scale
        if input_shape[-1] != 1:
            self.db2_lpf = tf.repeat(self.db2_lpf, input_shape[-1], axis=-1)
            self.db2_hpf = tf.repeat(self.db2_hpf, input_shape[-1], axis=-1)

    def call(self, inputs, training=None, mask=None):

        # border padding symatric add coulums
        inputs_pad = tf.pad(inputs, [[0, 0], [0, 0], [self.dec_len-1, self.dec_len-1], [0, 0]], self.border_padd)

        # approximation conv only rows
        a = tf.nn.conv2d(
            inputs_pad, self.db2_lpf, padding=self.conv_type, strides=[1, 1, 1, 1],
        )
        # details conv only rows
        d = tf.nn.conv2d(
            inputs_pad, self.db2_hpf, padding=self.conv_type, strides=[1, 1, 1, 1],
        )
        # ds - down sample
        a_ds = a[:, :, 1:a.shape[2]:2, :]
        d_ds = d[:, :, 1:d.shape[2]:2, :]

        # border padding symatric add rows
        a_ds_pad = tf.pad(a_ds, [[0, 0], [self.dec_len-1, self.dec_len-1], [0, 0], [0, 0]], self.border_padd)
        d_ds_pad = tf.pad(d_ds, [[0, 0], [self.dec_len-1, self.dec_len-1], [0, 0], [0, 0]], self.border_padd)

        # convolution is done on the rows so we need to
        # transpose the matrix in order to convolve the colums
        a_ds_pad = tf.transpose(a_ds_pad, perm=[0, 2, 1, 3])
        d_ds_pad = tf.transpose(d_ds_pad, perm=[0, 2, 1, 3])

        # aa approximation approximation
        aa = tf.nn.conv2d(
            a_ds_pad, self.db2_lpf, padding=self.conv_type, strides=[1, 1, 1, 1],
        )
        # ad approximation details
        ad = tf.nn.conv2d(
            a_ds_pad, self.db2_hpf, padding=self.conv_type, strides=[1, 1, 1, 1],
        )
        # ad details aproximation
        da = tf.nn.conv2d(
            d_ds_pad, self.db2_lpf, padding=self.conv_type, strides=[1, 1, 1, 1],
        )
        # dd details details
        dd = tf.nn.conv2d(
            d_ds_pad, self.db2_hpf, padding=self.conv_type, strides=[1, 1, 1, 1],
        )

        # transpose back the matrix
        aa = tf.transpose(aa, perm=[0, 2, 1, 3])
        ad = tf.transpose(ad, perm=[0, 2, 1, 3])
        da = tf.transpose(da, perm=[0, 2, 1, 3])
        dd = tf.transpose(dd, perm=[0, 2, 1, 3])

        # down sample
        ll = aa[:, 1:aa.shape[1]:2, :, :]
        lh = ad[:, 1:ad.shape[1]:2, :, :]
        hl = da[:, 1:da.shape[1]:2, :, :]
        hh = dd[:, 1:dd.shape[1]:2, :, :]

        # concate all outputs ionto tensor
        x = tf.concat([ll, lh, hl, hh], axis=-1)

        return x


class IDWT(layers.Layer):
    def __init__(self, name='haar', **kwargs):
        super(IDWT, self).__init__(**kwargs)
        self._name = self.name + "_" + name
        self.pad_type = "VALID"
        self.border_pad = "SYMMETRIC"

        # get filter coeffs from 3rd party lib
        wavelet = pywt.Wavelet(name)
        self.rec_len = wavelet.rec_len

        # decomposition filter low pass and hight pass coeffs
        db2_lpf = wavelet.rec_lo
        db2_hpf = wavelet.rec_hi

        # covert filters into tensors and reshape for convolution math
        db2_lpf = tf.constant(db2_lpf[::-1])
        self.db2_lpf = tf.reshape(db2_lpf, (1, wavelet.rec_len, 1, 1))

        db2_hpf = tf.constant(db2_hpf[::-1])
        self.db2_hpf = tf.reshape(db2_hpf, (1, wavelet.rec_len, 1, 1))

    def upsampler2d(self, x):
        """
        up sampling with zero insertion between rows and columns
        :param x: 4 dim tensor (?, w, h, ch)
        :return:  up sampled tensor with shape (?, 2*w, 2*h, ch)
        """
        # create zero like tensor
        zero_tensor = tf.zeros_like(x)
        # stack both tensors
        stack_rows = tf.stack([x, zero_tensor], axis=3)
        # reshape for zero insertion between the rows
        stack_rows = tf.reshape(stack_rows, shape=[-1, x.shape[1], x.shape[2]*2, x.shape[3]])
        # transpose in order to insert zeros for the columns
        stack_rows = tf.transpose(stack_rows, perm=[0, 2, 1, 3])
        # create zero like tensor but now like the padded one
        zero_tensor_1 = tf.zeros_like(stack_rows)
        # stack both tensors
        stack_rows_cols = tf.stack([stack_rows, zero_tensor_1], axis=3)
        # reshape for zero insertion between the columns
        us_padded = tf.reshape(stack_rows_cols, shape=[-1, x.shape[1]*2, x.shape[2]*2, x.shape[3]])
        # transpose back to normal
        us_padded = tf.transpose(us_padded, perm=[0, 2, 1, 3])
        return us_padded

    def call(self, inputs, training=None, mask=None):

        # border padding for convolution with low pass and high pass filters
        x = tf.pad(inputs,
                   [[0, 0], [self.rec_len-1, self.rec_len-1], [self.rec_len-1, self.rec_len-1], [0, 0]],
                   self.border_pad)

        # convert to float32
        # x = tf.cast(x, tf.float32)
        # GPU works with float 32
        # CPU  can work with 64 but need to add extra flag
        # convert to float64
        # x = tf.cast(x, tf.float64)

        # extract approximation and details from input tensor
        # TODO: whit if tensor shape is bigger then 4?
        # and expand the dims for the up sampling
        ll = tf.expand_dims(x[:, :, :, 0], axis=-1)
        lh = tf.expand_dims(x[:, :, :, 1], axis=-1)
        hl = tf.expand_dims(x[:, :, :, 2], axis=-1)
        hh = tf.expand_dims(x[:, :, :, 3], axis=-1)

        ll_us_pad = self.upsampler2d(ll)
        lh_us_pad = self.upsampler2d(lh)
        hl_us_pad = self.upsampler2d(hl)
        hh_us_pad = self.upsampler2d(hh)

        # convolution for the rows
        # transpose for the column convolution
        # convolution for the column
        # transpose back to normal

        ll_conv_lpf = tf.nn.conv2d(ll_us_pad, self.db2_lpf, padding=self.pad_type, strides=[1, 1, 1, 1], )
        ll_conv_lpf_tr = tf.transpose(ll_conv_lpf, perm=[0, 2, 1, 3])
        ll_conv_lpf_lpf = tf.nn.conv2d(ll_conv_lpf_tr, self.db2_lpf, padding=self.pad_type, strides=[1, 1, 1, 1], )
        ll_conv_lpf_lpf_tr = tf.transpose(ll_conv_lpf_lpf, perm=[0, 2, 1, 3])

        lh_conv_lpf = tf.nn.conv2d(lh_us_pad, self.db2_lpf, padding=self.pad_type, strides=[1, 1, 1, 1], )
        lh_conv_lpf_tr = tf.transpose(lh_conv_lpf, perm=[0, 2, 1, 3])
        lh_conv_lpf_hpf = tf.nn.conv2d(lh_conv_lpf_tr, self.db2_lpf, padding=self.pad_type, strides=[1, 1, 1, 1], )
        lh_conv_lpf_hpf_tr = tf.transpose(lh_conv_lpf_hpf, perm=[0, 2, 1, 3])

        hl_conv_hpf = tf.nn.conv2d(hl_us_pad, self.db2_hpf, padding=self.pad_type, strides=[1, 1, 1, 1], )
        hl_conv_hpf_tr = tf.transpose(hl_conv_hpf, perm=[0, 2, 1, 3])
        hl_conv_hpf_lpf = tf.nn.conv2d(hl_conv_hpf_tr, self.db2_lpf, padding=self.pad_type, strides=[1, 1, 1, 1], )
        hl_conv_hpf_lpf_tr = tf.transpose(hl_conv_hpf_lpf, perm=[0, 2, 1, 3])

        hh_conv_hpf = tf.nn.conv2d(hh_us_pad, self.db2_hpf, padding=self.pad_type, strides=[1, 1, 1, 1], )
        hh_conv_hpf_tr = tf.transpose(hh_conv_hpf, perm=[0, 2, 1, 3])
        hh_conv_hpf_hpf = tf.nn.conv2d(hh_conv_hpf_tr, self.db2_hpf, padding=self.pad_type, strides=[1, 1, 1, 1], )
        hh_conv_hpf_hpf_tr = tf.transpose(hh_conv_hpf_hpf, perm=[0, 2, 1, 3])

        # add all together
        reconstructed = tf.add_n([ll_conv_lpf_lpf_tr,
                                  lh_conv_lpf_hpf_tr,
                                  hl_conv_hpf_lpf_tr,
                                  hh_conv_hpf_hpf_tr])
        # crop the paded part
        crop = (self.rec_len -1)*2
        return reconstructed[:, crop-1:-crop, crop-1:-crop, :]


if __name__ == "__main__":
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # x_train = x_train.astype("float32")
    # x_test = x_test.astype("float32")
    # # x_train = cv2.imread("../input/LennaGrey.png", 0)
    # frog = tf.expand_dims(
    #     x_train[0, :, :, :], 0, name=None
    # )
    # print("frog shape", frog.shape)
    # model = keras.Sequential()
    # model.add(keras.Input(shape=(256, 256, 4)))
    # model.add(IDWT())
    # model.summary()

    name = "db2"
    img = cv2.imread("../input/LennaGrey.png",0)
    img_ex1 = np.expand_dims(img, axis=-1)
    img_ex2 = np.expand_dims(img_ex1, axis=0)

    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=img_ex1.shape))
    model.add(DWT(name=name))
    # model.summary()
    coeffs = model.predict(img_ex2)
    LL = coeffs[0, ..., 0]
    LH = coeffs[0, ..., 1]
    HL = coeffs[0, ..., 2]
    HH = coeffs[0, ..., 3]

    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=coeffs[0].shape))
    model.add(IDWT(name=name))
    model.summary()

    my_recon = model.predict(coeffs)
    img_my_rec = my_recon[0, :, :, 0]
    coeffs2 = pywt.wavedec2(img, name,level=1)

    LL2 = coeffs2[0]
    LH2 = coeffs2[1][0]
    HL2 = coeffs2[1][1]
    HH2 = coeffs2[1][2]

    recon_pywt = pywt.waverec2(coeffs2, name)
    img_pywt_rec = recon_pywt

    print("LL mse ", mse.mse(LL, LL2))
    print("LH mse ", mse.mse(LH, LH2))
    print("HL mse ", mse.mse(HL, HL2))
    print("HH mse ", mse.mse(HH, HH2))
    print("img mse ", mse.mse(img_pywt_rec, img_my_rec))

    difference = cv2.absdiff(np.int32(img_my_rec), np.int32(img_pywt_rec))
    _, mask = cv2.threshold(difference.astype("uint8"), 0, 255, cv2.THRESH_BINARY)

    cv2.imshow("diff", mask)
    cv2.waitKey(0)
    pass





    # a = model.predict(frog, steps=1)
    # #
    # approx = tf.image.convert_image_dtype(a[0, ..., 0], dtype=tf.float32)
    # with tf.Session() as sess:
    #     img = sess.run(approx)
    # #     pass
    # #
    # img = np.clip(img, 0, 255)
    # img = np.ceil(img)
    # img = img.astype("uint8")
    # with open(r"D:\TEMP\LL_python_layer.raw", "wb") as outfile:
    #     outfile.write(img)  # Write it

    # model = models.WaveletCifar10CNN.WaveletCNN((32,32,3), 10)
    # model.summary()