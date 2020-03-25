from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def CEN_model(image_size):
    """
    CNN for semgentation of multplie sclerosis lesion.
    This architecture was inspired by paper of Brosch et al. 
    "Deep Convolutional Encoder Networks for Multiple Sclerosis Lesion Segmentation"
    """
    model = Sequential()
    model.add(Conv3D(filters=32,
                kernel_size=(9, 9, 9),
                name="conv1",
                activation=None,    
                padding="valid",
                input_shape=image_size))
    model.add(ELU(alpha=1.0))
    model.add(Conv3DTranspose(filters=1,
                kernel_size=(9, 9, 9),
                name="Tconv2",
                activation="sigmoid"))
    model.summary()
    return model

def get_3Dconv_layer(layer_in, num_filters, dropout_rate=0.2, kernel_size=(3, 3, 3), strides=1, activation='relu'):
    layer = BatchNormalization(axis=1)(layer_in)
    layer = Conv3D(num_filters, kernel_size=kernel_size, activation=activation, padding='same')(layer)
    layer_out = Dropout(dropout_rate)(layer)
    return layer_out

def get_3Ddownconvolution_layer(layer_in, num_filters, dropout_rate=0.0, kernel_size=(3, 3, 3), activation='relu', strides=(1, 1, 1)):
    layer = BatchNormalization(axis=1)(layer_in)
    layer = Conv3D(num_filters, kernel_size=kernel_size, activation=activation, padding='same', strides=strides)(layer)
    layer_out = Dropout(dropout_rate)(layer)
    return layer_out

def get_downconvolution_1block(input_layer, filters, dropout_rate):
    
    l1 = get_3Dconv_layer(input_layer, 
                            filters, 
                            dropout_rate=dropout_rate, 
                            kernel_size=(5, 5, 5), 
                            activation=None)
    l2 = PReLU(shared_axes=(2,3,4))(l1)
    l3 = Add()([l2, input_layer])
    l4 = get_3Ddownconvolution_layer(l3, 
                                     2*filters, 
                                     dropout_rate=0.0, 
                                     kernel_size=(2, 2, 2), 
                                     activation=None, 
                                     strides=(2, 2, 2))
    l5 = PReLU(shared_axes=(2,3,4))(l4)
    
    return l5, l3

def get_downconvolution_2blocks(input_layer, filters, dropout_rate):
    
    l1 = get_3Dconv_layer(input_layer, 
                            filters, 
                            dropout_rate=dropout_rate, 
                            kernel_size=(5, 5, 5), 
                            activation=None)
    l2 = PReLU(shared_axes=(2,3,4))(l1)
    l3 = get_3Dconv_layer(l2, 
                          filters, 
                          dropout_rate=dropout_rate, 
                          kernel_size=(5, 5, 5), 
                          activation=None)
    l4 = PReLU(shared_axes=(2,3,4))(l3)
    l5 = Add()([l4, input_layer])
    l6 = get_3Ddownconvolution_layer(l5, 
                                     2*filters, 
                                     dropout_rate=0.0, 
                                     kernel_size=(2, 2, 2), 
                                     activation=None, 
                                     strides=(2, 2, 2))
    l7 = PReLU(shared_axes=(2,3,4))(l6)
    
    return l7, l5

def get_downconvolution_3blocks(input_layer, filters, dropout_rate):
    
    l1 = get_3Dconv_layer(input_layer, 
                            filters, 
                            dropout_rate=dropout_rate, 
                            kernel_size=(5, 5, 5), 
                            activation=None)
    l2 = PReLU(shared_axes=(2,3,4))(l1)
    l3 = get_3Dconv_layer(l2, 
                          filters, 
                          dropout_rate=dropout_rate, 
                          kernel_size=(5, 5, 5), 
                          activation=None)
    l4 = PReLU(shared_axes=(2,3,4))(l3)
    l5 = get_3Dconv_layer(l4, 
                          filters, 
                          dropout_rate=dropout_rate, 
                          kernel_size=(5, 5, 5), 
                          activation=None)
    l6 = PReLU(shared_axes=(2,3,4))(l5)
    l7 = Add()([l6, input_layer])
    l8 = get_3Ddownconvolution_layer(l7, 
                                     2*filters, 
                                     dropout_rate=0.0, 
                                     kernel_size=(2, 2, 2), 
                                     activation=None, 
                                     strides=(2, 2, 2))
    l9 = PReLU(shared_axes=(2,3,4))(l8)
    
    return l9, l7

def get_3Dupconvolution_layer(layer_in, num_filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation=None):
    layer_out = Conv3DTranspose(num_filters, kernel_size=kernel_size, strides=strides, padding='same', activation=activation)(layer_in)
    return layer_out

def get_last_convolution_block(input_layer, filters, dropout_rate):

    l1 = get_3Dconv_layer(input_layer, 
                            filters, 
                            dropout_rate=dropout_rate, 
                            kernel_size=(5, 5, 5), 
                            activation=None)
    l2 = PReLU(shared_axes=(2,3,4))(l1)
    l3 = get_3Dconv_layer(l2, 
                          filters, 
                          dropout_rate=dropout_rate, 
                          kernel_size=(5, 5, 5), 
                          activation=None)
    l4 = PReLU(shared_axes=(2,3,4))(l3)
    l5 = get_3Dconv_layer(l4, 
                          filters, 
                          dropout_rate=dropout_rate, 
                          kernel_size=(5, 5, 5), 
                          activation=None)
    l6 = PReLU(shared_axes=(2,3,4))(l5)
    l7 = Add()([l6, input_layer])
    l8 = get_3Dupconvolution_layer(l7, 
                                   filters//2, 
                                   kernel_size=(2, 2, 2), 
                                   strides=(2, 2, 2), 
                                   activation=None)
    l9 = PReLU(shared_axes=(2,3,4))(l8)
    
    return l9

def get_upconvolution_3blocks(input_layer, res_layer, filters, dropout_rate):
    
    r1 = Concatenate()([input_layer, res_layer])
    r2 = get_3Dconv_layer(r1, 
                            filters, 
                            dropout_rate=dropout_rate, 
                            kernel_size=(5, 5, 5), 
                            activation=None)
    r3 = PReLU(shared_axes=(2,3,4))(r2)
    r4 = get_3Dconv_layer(r3, 
                            filters, 
                            dropout_rate=dropout_rate, 
                            kernel_size=(5, 5, 5), 
                            activation=None)
    r5 = PReLU(shared_axes=(2,3,4))(r4)
    r6 = get_3Dconv_layer(r5, 
                            filters, 
                            dropout_rate=dropout_rate, 
                            kernel_size=(5, 5, 5), 
                            activation=None)
    r7 = PReLU(shared_axes=(2,3,4))(r6)
    r8 = Add()([r7, input_layer])
    r9 = get_3Dupconvolution_layer(r8, 
                                   filters//2, 
                                   kernel_size=(2, 2, 2), 
                                   strides=(2, 2, 2), 
                                   activation=None)
    r10 = PReLU(shared_axes=(2,3,4))(r9)
        
    return r10

def get_upconvolution_2blocks(input_layer, res_layer, filters, dropout_rate):
    
    r1 = Concatenate()([input_layer, res_layer])
    r2 = get_3Dconv_layer(r1, 
                            filters, 
                            dropout_rate=dropout_rate, 
                            kernel_size=(5, 5, 5), 
                            activation=None)
    r3 = PReLU(shared_axes=(2,3,4))(r2)
    r4 = get_3Dconv_layer(r3, 
                            filters, 
                            dropout_rate=dropout_rate, 
                            kernel_size=(5, 5, 5), 
                            activation=None)
    r5 = PReLU(shared_axes=(2,3,4))(r4)
    r6 = Add()([r5, input_layer])
    r7 = get_3Dupconvolution_layer(r6, 
                                   filters//2, 
                                   kernel_size=(2, 2, 2), 
                                   strides=(2, 2, 2), 
                                   activation=None)
    r8 = PReLU(shared_axes=(2,3,4))(r7)
        
    return r8

def get_upconvolution_1block(input_layer, res_layer, filters, dropout_rate):
    
    r1 = Concatenate()([input_layer, res_layer])
    r2 = get_3Dconv_layer(r1, 
                            filters, 
                            dropout_rate=dropout_rate, 
                            kernel_size=(5, 5, 5), 
                            activation=None)
    r3 = PReLU(shared_axes=(2,3,4))(r2)
    r4 = Add()([r3, input_layer])
    r5 = get_3Dconv_layer(r4, 
                            1, 
                            dropout_rate=dropout_rate, 
                            kernel_size=(5, 5, 5), 
                            activation=None)
    r6 = PReLU(shared_axes=(2,3,4))(r5)
        
    return r6


def Vnet_model(modalities, patch_size, filters=16, dropout_rate=0.2):
    channels = modalities
    input_layer = Input(shape=patch_size)
    
    l1, l1_3 = get_downconvolution_1block(input_layer, 1*filters, dropout_rate)

    l2, l2_5 = get_downconvolution_2blocks(l1, 2*filters, dropout_rate)
    
    l3, l3_7 = get_downconvolution_3blocks(l2, 4*filters, dropout_rate)
    
    l4, l4_7 = get_downconvolution_3blocks(l3, 8*filters, dropout_rate)

    l5 = get_last_convolution_block(l4, 16*filters, dropout_rate)
    
    r4 = get_upconvolution_3blocks(l5, l4_7, 8*filters, dropout_rate)
    
    r3 = get_upconvolution_3blocks(r4, l3_7, 4*filters, dropout_rate)
    
    r2 = get_upconvolution_2blocks(r3, l2_5, 2*filters, dropout_rate)

    r1 = get_upconvolution_1block(r2, l1_3, 1*filters, dropout_rate)
    
    output_layer = Softmax()(r1)

    model = Model(inputs=[input_layer], outputs=[output_layer])

    return model
