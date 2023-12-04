import numpy as np
import tensorflow as tf
from tensorflow.python.keras import losses


from sklearn.model_selection import train_test_split


from keras.models import Model
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Concatenate, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Lambda, Dropout
from keras.optimizers import Adam


import cv2
import glob


size = 768
num_img = 50


img_list = glob.glob('/Work_data/Image/*')
mask_list = glob.glob('/Work_data/Mask/*')

img_list.sort()
mask_list.sort()

img_list = img_list[0:num_img]
mask_list = mask_list[0:num_img]


images_data = []
for name in img_list:
    images_data.append(cv2.imread(name, 0))

images_data = np.array(images_data)
images_data = np.expand_dims(images_data, axis = 3)


mask_data = []
for name in mask_list:
    mask_data.append(cv2.imread(name, 0))

mask_data = np.array(mask_data)
mask_data = np.expand_dims(mask_data, axis = 3)


images_data = images_data/ 225
mask_data = mask_data / 255






X_train, X_test, y_train, y_test = train_test_split(images_data, mask_data, test_size=0.25, random_state=1)



def conv_block(input, filter):

    x = Conv2D(filter, 3, padding = 'same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filter, 3, padding = 'same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return(x)

def encoder_block(input, filter):

    x = conv_block(input, filter)
    #p = MaxPooling2D(2, 2)(x)   
    p = MaxPooling2D((2,2), padding='same')(x)

    return x, p

def decoder_block(input, skip, filter):

    x = Conv2DTranspose(filter, (2, 2), strides = 2, padding = "same")(input)
    x = Concatenate()([x, skip])
    x = conv_block(x, filter)

    return x



def unet(input_shape, classes):

    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)


    b1 = conv_block(p4, 1024)


    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    output = Conv2D(classes, 1, padding = 'same', activation = 'sigmoid')(d4)

    model = Model(inputs, output, name="Unet")
    return model


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss



shape = (images_data.shape[1],images_data.shape[2], images_data.shape[3])

model = unet(shape, classes=1)
model.compile(optimizer = Adam(learning_rate= 1e-3), loss = bce_dice_loss, metrics=[dice_loss])
model.summary()


hiss = model.fit(X_train, y_train, batch_size=4, verbose =1, epochs=1, validation_data=(X_test, y_test), shuffle=False)

model.save('/Model_list/unet_test.keras')