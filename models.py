from tensorflow.keras.layers import Dense, Input,BatchNormalization,LayerNormalization,Dropout,Conv2D,Conv2DTranspose, MaxPooling2D, Flatten,Activation, PReLU,concatenate, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation



def PrecipNet(input_shape,activation='elu', optimizer='adam', up_convT=True, version='regression'):
    '''
    activation : relu or elu
    opt : optimizer
    up_convT : {True: upsampling method using Conv2DTranspose layer, False:using Upsampleing layer}
    '''
    
    drop = 0.5
    unit1 = 64
    unit2 = 128
    unit3 = 256
    unit4 = 512
    unit5 = 1024

    inputs = Input(shape=(input_shape))
    inputs_ = UpSampling2D(size=(2,2))(inputs)
    conv1 = Conv2D(unit1, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs_)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation)(conv1)
    conv1 = Conv2D(unit1, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(drop)(pool1)
    conv2 = Conv2D(unit2, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation)(conv2)
    conv2 = Conv2D(unit2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(drop)(pool2)
    conv3 = Conv2D(unit3, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation(activation)(conv3)
    conv3 = Conv2D(unit3, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation(activation)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(drop)(pool3)
    conv4 = Conv2D(unit4, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation(activation)(conv4)
    conv4 = Conv2D(unit4, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation(activation)(conv4)
    drop4 = Dropout(drop)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(unit5, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation(activation)(conv5)
    conv5 = Conv2D(unit5, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation(activation)(conv5)
    drop5 = Dropout(drop)(conv5)
    
    if up_convT:
        up6 = Conv2D(unit4, 2, padding = 'same', kernel_initializer = 'he_normal')(Conv2DTranspose(unit4,2, activation='relu',padding='same',  strides = (2,2))(drop5))
    else: 
        up6 = Conv2D(unit4, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        
    up6 = BatchNormalization()(up6)
    up6 = Activation(activation)(up6)
    merge6 = concatenate([drop4,up6], axis = 3)
    merge6 = Dropout(drop)(merge6)
    conv6 = Conv2D(unit4, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation(activation)(conv6)
    conv6 = Conv2D(unit4, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation(activation)(conv6)
    
    if up_convT:
        up7 = Conv2D(unit3, 2, padding = 'same', kernel_initializer = 'he_normal')(Conv2DTranspose(unit3,2, activation='relu',padding='same',  strides = (2,2))(conv6))
    else:
        up7 = Conv2D(unit3, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = BatchNormalization()(up7)
    up7 = Activation(activation)(up7)
    merge7 = concatenate([conv3,up7], axis = 3)
    merge7 = Dropout(drop)(merge7)
    conv7 = Conv2D(unit3, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation(activation)(conv7)
    conv7 = Conv2D(unit3, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation(activation)(conv7)
    
    if up_convT:
        up8 = Conv2D(unit2, 2, padding = 'same', kernel_initializer = 'he_normal')(Conv2DTranspose(unit2,2, activation='relu',padding='same',  strides = (2,2))(conv7))
    else:
        up8 = Conv2D(unit2, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    up8 = BatchNormalization()(up8)
    up8 = Activation(activation)(up8)
    merge8 = concatenate([conv2,up8], axis = 3)
    merge8 = Dropout(drop)(merge8)
    conv8 = Conv2D(unit2, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation(activation)(conv8)
    conv8 = Conv2D(unit2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation(activation)(conv8)

    if up_convT:
        up9 = Conv2D(unit1, 2, padding = 'same', kernel_initializer = 'he_normal')(Conv2DTranspose(unit1,2, activation='relu',padding='same',  strides = (2,2))(conv8))
    else:
        up9 = Conv2D(unit1, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9 = BatchNormalization()(up9)
    up9 = Activation(activation)(up9)
    merge9 = concatenate([conv1,up9], axis = 3)
    merge9 = Dropout(drop)(merge9)
    conv9 = Conv2D(unit1, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation(activation)(conv9)
    conv9 = Conv2D(unit1, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation(activation)(conv9)
    conv9 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9) 
    conv9 = Activation(activation)(conv9)
    conv9 = MaxPooling2D(pool_size=(2, 2))(conv9)
    conv9 = BatchNormalization()(conv9)

    if version=='regression':
        regression = Conv2D(1, 1, activation = 'relu', name='regression')(conv9)
        model = Model(inputs, regression)
        model.compile(loss='mae', optimizer=optimizer)
    elif version=='classification':
        classification = Conv2D(1, 1, activation = 'sigmoid', name='classification')(conv9)
        model = Model(inputs, classification)
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return model
    