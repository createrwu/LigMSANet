from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, concatenate, add, UpSampling2D, DepthwiseConv2D
from keras.models import Model
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout, Activation, BatchNormalization, ZeroPadding2D, Softmax
from keras.layers import Activation, BatchNormalization, add, Reshape, Add, multiply
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.applications.mobilenet_v2 import mobilenet_v2
import keras
from tensorflow.keras import backend
from keras.utils.data_utils import get_file

from keras.layers.core import Lambda
import tensorflow as tf
from tensorflow.keras import layers





def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', use_activation=True):
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    if use_activation:
        x = Activation('relu')(x)
        return x
    else:
        return x

def relu6(x):
    return K.relu(x, max_value=6)

def correct_pad(inputs, kernel_size):
    img_dim = 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    in_channels = backend.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'mobl{}_'.format(block_id)+'conv_{}_'.format(block_id)
    prefix1 = 'bn{}_'.format(block_id)+'conv_{}_'.format(block_id)

    if block_id:
        # Expand
        x = Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None
                        )(x)
        x = BatchNormalization(epsilon=1e-3,momentum=0.999 )(x)
        x = Activation(relu6)(x)
    else:
        prefix = prefix = 'expanded_conv_'

    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(x, 3))(x)

    x = DepthwiseConv2D(kernel_size=3,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding='same' if stride == 1 else 'valid'
                               )(x) 
    x = BatchNormalization(epsilon=1e-3,
                           momentum=0.999
                           )(x)

    x = Activation(relu6)(x)


    x = Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None)(x)

    x = BatchNormalization(epsilon=1e-3,
                                  momentum=0.999)(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add()([inputs, x])
    return x 


def LigMSANet(input_shape=(None, None, 3),BN=True,alpha=1.0, include_top=False, weights='imagenet'):
    rows = 224    
    input_flow = Input(shape=input_shape)    
    dilated_conv_kernel_initializer = RandomNormal(stddev=0.01)
    conv_kernel_initializer = RandomNormal(stddev=0.01)
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = ZeroPadding2D(padding=correct_pad(input_flow, 3),name='Conv1_pad')(input_flow)
    x = Conv2D(first_block_filters,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,name='Conv1')(x)
    x = BatchNormalization(epsilon=1e-3,momentum=0.999,name='bn_Conv1')(x)
    x = Activation(relu6,name='Conv1_relu')(x)
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,expansion=1, block_id=0)

    # 200,200,16 -> 100,100,24
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,expansion=6, block_id=2)

    # 100,100,24 -> 50,50,32
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,expansion=6, block_id=5)
    
    # 50,50,32 -> 25,25,64
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,expansion=6, block_id=9)

    # 25,25,64 -> 25,25,96
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,expansion=6, block_id=12)


    xs = []
    # branch 1
    x1_3 = DepthwiseConv2D(kernel_size = (3, 3), strides=(1, 1), padding='same',
                           depthwise_initializer=conv_kernel_initializer, use_bias=False, name='fpm1_3_dconv2d')(x)
    x1_3 = BatchNormalization(name='fpm1_3_dbn')(x1_3) if BN else x1_3
    x1_3 = Activation('relu', name='fpm1_3__drelu')(x1_3)
    x1_3 = Conv2D(32, (1,1), strides=(1, 1), padding='same',kernel_initializer=conv_kernel_initializer, use_bias=False, name='fpm1_3_conv2d')(x1_3)
    x1_3 = BatchNormalization( name='fpm1_3_bn')(x1_3) if BN else x1_3
    x1_3 = Activation('relu',  name='fpm1_3_relu')(x1_3)
    xs.append(x1_3)
    x1_3_1 = Lambda(lambda x: tf. keras.backend.expand_dims(x, axis=-1))(x1_3)
    
    # branch 2
    x1_5 = DepthwiseConv2D(kernel_size = (5, 5), strides=(1, 1), padding='same',
                           depthwise_initializer=conv_kernel_initializer, use_bias=False, name='fpm1_5_dconv2d')(x)
    x1_5 = BatchNormalization(name='fpm1_5_dbn')(x1_5) if BN else x1_5
    x1_5 = Activation('relu', name='fpm1_5_drelu')(x1_5)
    x1_5 = Conv2D(32, (1,1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer, use_bias=False,name='fpm1_5_conv2d')(x1_5)
    x1_5 = BatchNormalization( name='fpm1_5_bn')(x1_5) if BN else x1_5
    x1_5 = Activation('relu',  name='fpm1_5_relu')(x1_5)
    x1_3_5 = concatenate([x1_3, x1_5])
    x1_3_5 = DepthwiseConv2D(kernel_size = (3, 3), strides=(1, 1), padding='same',
                             depthwise_initializer=conv_kernel_initializer, use_bias=False, name='fpm1_3_5_dconv2d')(x1_3_5)
    x1_3_5 = BatchNormalization(name='fpm1_3_5_dbn')(x1_3_5) if BN else x1_3_5
    x1_3_5 = Activation('relu', name='fpm1_3_5_drelu')(x1_3_5)
    x1_3_5 = Conv2D(32, (1,1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer, use_bias=False,name='fpm1_3_5_conv2d')(x1_3_5)
    x1_3_5 = BatchNormalization( name='fpm1_3_5_bn')(x1_3_5) if BN else x1_3_5
    x1_3_5 = Activation('relu',  name='fpm1_3_5_relu')(x1_3_5)
    xs.append(x1_3_5)
    x1_3_5_1 = Lambda(lambda x: tf. keras.backend.expand_dims(x, axis=-1))(x1_3_5)
    
    # branch 3
    x1_7 = DepthwiseConv2D(kernel_size = (7, 7), strides=(1, 1), padding='same', 
                           depthwise_initializer=conv_kernel_initializer, use_bias=False, name='fpm1_7_dconv2d')(x)
    x1_7 = BatchNormalization(name='fpm1_7_dbn')(x1_7) if BN else x1_7
    x1_7 = Activation('relu', name='fpm1_7_drelu')(x1_7)
    x1_7 = Conv2D(32, (1,1), strides=(1, 1), padding='same',kernel_initializer=conv_kernel_initializer, use_bias=False,name='fpm1_7_conv2d')(x1_7)
    x1_7 = BatchNormalization( name='fpm1_7_bn')(x1_7) if BN else x1_7
    x1_7 = Activation('relu',  name='fpm1_7_relu')(x1_7)
    x1_3_5_7 = concatenate([x1_7, x1_3_5])
    x1_3_5_7 = DepthwiseConv2D(kernel_size = (3, 3), strides=(1, 1), padding='same',
                               depthwise_initializer=conv_kernel_initializer, use_bias=False, name='fpm1_3_5_7_dconv2d')(x1_3_5_7)
    x1_3_5_7 = BatchNormalization(name='fpm1_3_5_7_dbn')(x1_3_5_7) if BN else x1_3_5_7
    x1_3_5_7 = Activation('relu', name='fpm1_3_5_7_drelu')(x1_3_5_7)
    x1_3_5_7 = Conv2D(32, (1,1), strides=(1, 1), padding='same',kernel_initializer=conv_kernel_initializer, use_bias=False, name='fpm1_3_5_7_conv2d')(x1_3_5_7)
    x1_3_5_7 = BatchNormalization( name='fpm1_3_5_7_bn')(x1_3_5_7) if BN else x1_3_5_7
    x1_3_5_7 = Activation('relu',  name='fpm1_3_5_7_relu')(x1_3_5_7)
    xs.append(x1_3_5_7)
    x1_3_5_7_1 = Lambda(lambda x: tf. keras.backend.expand_dims(x, axis=-1))(x1_3_5_7)
    
    # branch 4
    x1_9 = DepthwiseConv2D(kernel_size = (9, 9), strides=(1, 1), padding='same',
                           depthwise_initializer=conv_kernel_initializer, use_bias=False, name='fpm1_9_dconv2d')(x)
    x1_9 = BatchNormalization(name='fpm1_9_dbn')(x1_9) if BN else x1_9
    x1_9 = Activation('relu', name='fpm1_9_drelu')(x1_9)
    x1_9 = Conv2D(32, (1,1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer,use_bias=False, name='fpm1_9_conv2d')(x1_9)
    x1_9 = BatchNormalization( name='fpm1_9_bn')(x1_9) if BN else x1_9
    x1_9 = Activation('relu',  name='fpm1_9_relu')(x1_9)
    x1_3_5_7_9 = concatenate([x1_9, x1_3_5_7])
    x1_3_5_7_9 = DepthwiseConv2D(kernel_size = (3, 3), strides=(1, 1), padding='same',
                                  depthwise_initializer=conv_kernel_initializer, use_bias=False, name='fpm1_3_5_7_9_dconv2d')(x1_3_5_7_9)
    x1_3_5_7_9 = BatchNormalization(name='fpm1_3_5_7_9_dbn')(x1_3_5_7) if BN else x1_3_5_7_9
    x1_3_5_7_9 = Activation('relu', name='fpm1_3_5_7_9_drelu')(x1_3_5_7_9)
    x1_3_5_7_9 = Conv2D(32, (1,1), strides=(1, 1), padding='same', 
                        kernel_initializer=conv_kernel_initializer,use_bias=False, name='fpm1_3_5_7_9_conv2d')(x1_3_5_7_9)
    x1_3_5_7_9 = BatchNormalization( name='fpm1_3_5_7_9_bn')(x1_3_5_7) if BN else x1_3_5_7_9
    x1_3_5_7_9 = Activation('relu',  name='fpm1_3_5_7_9_relu')(x1_3_5_7_9)
    xs.append(x1_3_5_7_9)
    x1_3_5_7_9_1 = Lambda(lambda x: tf. keras.backend.expand_dims(x, axis=-1))(x1_3_5_7_9)

 

    x1_Add = Add(name='1_add')(xs)
    # concat
    x_cat = concatenate([x1_3_1, x1_3_5_1, x1_3_5_7_1, x1_3_5_7_9_1],axis=-1)

    x1_gam = GlobalAveragePooling2D()(x1_Add)
    x1_gam = Reshape((1,1,-1), name='1_7_reshape')(x1_gam)
    x1_gam = Conv2D(4, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer, name='gam1_1_conv2d')(x1_gam)
    x1_gam = BatchNormalization( name='gam1_1_bn')(x1_gam) if BN else x1_gam
    x1_gam = Activation('relu', name='gam1_1_relu')(x1_gam)
    fatt1_1 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer,name='gam1_1_sigmoid')(x1_gam)
    fatt1_2 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer,name='gam1_2_sigmoid')(x1_gam)   
    fatt1_3 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer,name='gam1_3_sigmoid')(x1_gam)
    fatt1_4 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer,name='gam1_4_sigmoid')(x1_gam)
    fatt1_cat = concatenate([fatt1_1, fatt1_2, fatt1_3, fatt1_4])#[?, ?, ?, 128]
    fatt1_cat_reshape = Reshape((1, 1,32,4), name='1_2_reshape')(fatt1_cat)  
    attention_softmax = Softmax(axis=-1)(fatt1_cat_reshape)
    F_output = multiply([attention_softmax,x_cat])


    attention = []
    
    def slice(x,index):
        return x[:, :, :, :, index]    
        
    feature_1 = Lambda(slice, arguments={'index':0})(F_output)
    attention.append(feature_1)
    
    feature_2 = Lambda(slice, arguments={'index':1})(F_output)
    attention.append(feature_2)
    
    feature_3 = Lambda(slice, arguments={'index':2})(F_output)
    attention.append(feature_3)
    
    feature_4 = Lambda(slice, arguments={'index':3})(F_output)
    attention.append(feature_4)
    
    feature = Add(name='2_add')(attention)
 
    x = concatenate([x,feature])

    # FPM
    xx = []
    # branch 1
    x1_3 = DepthwiseConv2D(kernel_size = (3, 3), strides=(1, 1), padding='same',
                           depthwise_initializer=conv_kernel_initializer, use_bias=False, name='fpm2_3_dconv2d')(x)
    x1_3 = BatchNormalization(name='fpm2_3_dbn')(x1_3) if BN else x1_3
    x1_3 = Activation('relu', name='fpm2_3__drelu')(x1_3)
    x1_3 = Conv2D(32, (1,1), strides=(1, 1), padding='same',kernel_initializer=conv_kernel_initializer, use_bias=False, name='fpm2_3_conv2d')(x1_3)
    x1_3 = BatchNormalization( name='fpm2_3_bn')(x1_3) if BN else x1_3
    x1_3 = Activation('relu',  name='fpm2_3_relu')(x1_3)
    xx.append(x1_3)
    x1_3_1 = Lambda(lambda x: tf. keras.backend.expand_dims(x, axis=-1))(x1_3)
    
    # branch 2
    x1_5 = DepthwiseConv2D(kernel_size = (5, 5), strides=(1, 1), padding='same',
                           depthwise_initializer=conv_kernel_initializer, use_bias=False, name='fpm2_5_dconv2d')(x)
    x1_5 = BatchNormalization(name='fpm2_5_dbn')(x1_5) if BN else x1_5
    x1_5 = Activation('relu', name='fpm2_5_drelu')(x1_5)
    x1_5 = Conv2D(32, (1,1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer, use_bias=False,name='fpm2_5_conv2d')(x1_5)
    x1_5 = BatchNormalization( name='fpm2_5_bn')(x1_5) if BN else x1_5
    x1_5 = Activation('relu',  name='fpm2_5_relu')(x1_5)
    x1_3_5 = concatenate([x1_3, x1_5])
    x1_3_5 = DepthwiseConv2D(kernel_size = (3, 3), strides=(1, 1), padding='same',
                             depthwise_initializer=conv_kernel_initializer, use_bias=False, name='fpm2_3_5_dconv2d')(x1_3_5)
    x1_3_5 = BatchNormalization(name='fpm2_3_5_dbn')(x1_3_5) if BN else x1_3_5
    x1_3_5 = Activation('relu', name='fpm2_3_5_drelu')(x1_3_5)
    x1_3_5 = Conv2D(32, (1,1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer, use_bias=False,name='fpm2_3_5_conv2d')(x1_3_5)
    x1_3_5 = BatchNormalization( name='fpm2_3_5_bn')(x1_3_5) if BN else x1_3_5
    x1_3_5 = Activation('relu',  name='fpm2_3_5_relu')(x1_3_5)
    xx.append(x1_3_5)
    x1_3_5_1 = Lambda(lambda x: tf. keras.backend.expand_dims(x, axis=-1))(x1_3_5)
    
    # branch 3
    x1_7 = DepthwiseConv2D(kernel_size = (7, 7), strides=(1, 1), padding='same', 
                           depthwise_initializer=conv_kernel_initializer, use_bias=False, name='fpm2_7_dconv2d')(x)
    x1_7 = BatchNormalization(name='fpm2_7_dbn')(x1_7) if BN else x1_7
    x1_7 = Activation('relu', name='fpm2_7_drelu')(x1_7)
    x1_7 = Conv2D(32, (1,1), strides=(1, 1), padding='same',kernel_initializer=conv_kernel_initializer, use_bias=False,name='fpm2_7_conv2d')(x1_7)
    x1_7 = BatchNormalization( name='fpm2_7_bn')(x1_7) if BN else x1_7
    x1_7 = Activation('relu',  name='fpm2_7_relu')(x1_7)
    x1_3_5_7 = concatenate([x1_7, x1_3_5])
    x1_3_5_7 = DepthwiseConv2D(kernel_size = (3, 3), strides=(1, 1), padding='same',
                               depthwise_initializer=conv_kernel_initializer, use_bias=False, name='fpm2_3_5_7_dconv2d')(x1_3_5_7)
    x1_3_5_7 = BatchNormalization(name='fpm2_3_5_7_dbn')(x1_3_5_7) if BN else x1_3_5_7
    x1_3_5_7 = Activation('relu', name='fpm2_3_5_7_drelu')(x1_3_5_7)
    x1_3_5_7 = Conv2D(32, (1,1), strides=(1, 1), padding='same',kernel_initializer=conv_kernel_initializer, use_bias=False, name='fpm2_3_5_7_conv2d')(x1_3_5_7)
    x1_3_5_7 = BatchNormalization( name='fpm2_3_5_7_bn')(x1_3_5_7) if BN else x1_3_5_7
    x1_3_5_7 = Activation('relu',  name='fpm2_3_5_7_relu')(x1_3_5_7)
    xx.append(x1_3_5_7)
    x1_3_5_7_1 = Lambda(lambda x: tf. keras.backend.expand_dims(x, axis=-1))(x1_3_5_7)
    
    # branch 4
    x1_9 = DepthwiseConv2D(kernel_size = (9, 9), strides=(1, 1), padding='same',
                           depthwise_initializer=conv_kernel_initializer, use_bias=False, name='fpm2_9_dconv2d')(x)
    x1_9 = BatchNormalization(name='fpm2_9_dbn')(x1_9) if BN else x1_9
    x1_9 = Activation('relu', name='fpm2_9_drelu')(x1_9)
    x1_9 = Conv2D(32, (1,1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer,use_bias=False, name='fpm2_9_conv2d')(x1_9)
    x1_9 = BatchNormalization( name='fpm2_9_bn')(x1_9) if BN else x1_9
    x1_9 = Activation('relu',  name='fpm2_9_relu')(x1_9)
    x1_3_5_7_9 = concatenate([x1_9, x1_3_5_7])
    x1_3_5_7_9 = DepthwiseConv2D(kernel_size = (3, 3), strides=(1, 1), padding='same',
                                  depthwise_initializer=conv_kernel_initializer, use_bias=False, name='fpm2_3_5_7_9_dconv2d')(x1_3_5_7_9)
    x1_3_5_7_9 = BatchNormalization(name='fpm2_3_5_7_9_dbn')(x1_3_5_7) if BN else x1_3_5_7_9
    x1_3_5_7_9 = Activation('relu', name='fpm2_3_5_7_9_drelu')(x1_3_5_7_9)
    x1_3_5_7_9 = Conv2D(32, (1,1), strides=(1, 1), padding='same', 
                        kernel_initializer=conv_kernel_initializer,use_bias=False, name='fpm2_3_5_7_9_conv2d')(x1_3_5_7_9)
    x1_3_5_7_9 = BatchNormalization( name='fpm2_3_5_7_9_bn')(x1_3_5_7) if BN else x1_3_5_7_9
    x1_3_5_7_9 = Activation('relu',  name='fpm2_3_5_7_9_relu')(x1_3_5_7_9)
    xx.append(x1_3_5_7_9)
    x1_3_5_7_9_1 = Lambda(lambda x: tf. keras.backend.expand_dims(x, axis=-1))(x1_3_5_7_9)
    

 
    #Add
    x2_Add = Add(name='3_add')(xx)#[?, ?, ?, 32]
    # concat
    x_cat = concatenate([x1_3_1, x1_3_5_1, x1_3_5_7_1, x1_3_5_7_9_1],axis=-1)
    #GAM
    x1_gam = GlobalAveragePooling2D()(x2_Add)
    x1_gam = Reshape((1,1,-1), name='2_7_reshape')(x1_gam)
    x1_gam = Conv2D(4, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer, name='gam2_1_conv2d')(x1_gam)
    x1_gam = BatchNormalization( name='gam2_1_bn')(x1_gam) if BN else x1_gam
    x1_gam = Activation('relu', name='gam2_1_relu')(x1_gam)
    fatt1_1 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer,name='gam2_1_sigmoid')(x1_gam)
    fatt1_2 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer,name='gam2_2_sigmoid')(x1_gam)   
    fatt1_3 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer,name='gam2_3_sigmoid')(x1_gam)
    fatt1_4 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer,name='gam2_4_sigmoid')(x1_gam)
    fatt1_cat = concatenate([fatt1_1, fatt1_2, fatt1_3, fatt1_4])
    fatt1_cat_reshape = Reshape((1, 1,32,4), name='2_2_reshape')(fatt1_cat)  
    attention_softmax = Softmax(axis=-1)(fatt1_cat_reshape)
    F_output = multiply([attention_softmax,x_cat])
    

    attention2 = []
    
    def slice(x,index):
        return x[:, :, :, :, index]    
        
    feature_1 = Lambda(slice, arguments={'index':0})(F_output)
    attention2.append(feature_1)
    
    feature_2 = Lambda(slice, arguments={'index':1})(F_output)
    attention2.append(feature_2)
    
    feature_3 = Lambda(slice, arguments={'index':2})(F_output)
    attention2.append(feature_3)
    
    feature_4 = Lambda(slice, arguments={'index':3})(F_output)
    attention2.append(feature_4)
    
    feature = Add(name='4_add')(attention2)
 
    x = concatenate([x,feature])

 
    x = DepthwiseConv2D(kernel_size = (3, 3), strides=(1, 1), padding='same',
                           depthwise_initializer=conv_kernel_initializer, use_bias=False, name='1_dconv2d')(x)
    x = BatchNormalization(name='1_dbn')(x) if BN else x
    x = Activation('relu', name='1_drelu')(x)
    x = Conv2D(64, (1,1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer, use_bias=False,name='1_conv2d')(x)
    x = BatchNormalization( name='1_bn')(x) if BN else x
    x = Activation('relu',  name='1_relu')(x)
    x = DepthwiseConv2D(kernel_size = (3, 3), strides=(1, 1), padding='same',
                           depthwise_initializer=conv_kernel_initializer, use_bias=False, name='2_dconv2d')(x)
    x = BatchNormalization(name='2_dbn')(x) if BN else x
    x = Activation('relu', name='2_drelu')(x)
    x = Conv2D(32, (1,1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer, use_bias=False,name='2_conv2d')(x)
    x = BatchNormalization( name='2_bn')(x) if BN else x
    x = Activation('relu',  name='2_relu')(x)
    x = DepthwiseConv2D(kernel_size = (3, 3), strides=(1, 1), padding='same',
                           depthwise_initializer=conv_kernel_initializer, use_bias=False, name='3_dconv2d')(x)
    x = BatchNormalization(name='3_dbn')(x) if BN else x
    x = Activation('relu', name='3_drelu')(x)
    x = Conv2D(16, (1,1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer, use_bias=False,name='3_conv2d')(x)
    x = BatchNormalization( name='3_bn')(x) if BN else x
    x = Activation('relu',  name='3_relu')(x) 

    output_flow = Conv2D(1, 1, strides=(1, 1), padding='same', activation='relu', kernel_initializer=conv_kernel_initializer,name='1_1_Conv1')(x)
    model = Model(inputs=input_flow, outputs=output_flow)
    

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' +
                          str(alpha) + '_' + str(rows) + '.h5')
            weight_path = ('.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5')
            weights_path = get_file(
                model_name, weight_path, cache_subdir='models')
        else:
            model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' +
                          str(alpha) + '_' + str(rows) + '_no_top' + '.h5')
            weight_path = ('.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top1.h5')
            weights_path = get_file(
                model_name, weight_path, cache_subdir='models')
        model.load_weights('.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top1.h5',by_name = True)
    elif weights is not None:
        model.load_weights(weights)
    return model
