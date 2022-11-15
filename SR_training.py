import os
import tensorflow as tf
import numpy as np
import PIL
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import add,Dense
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation,Flatten
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from IPython.display import display
from skimage.transform import rescale, resize
# import georasters as gr
from matplotlib import image
from matplotlib import colors
import glob
from tqdm import tqdm
import imageio
# import rasterio
import argparse
# from rasterio.plot import reshape_as_image
from skimage import img_as_ubyte
import random
from PIL import Image
from keras.models import load_model


os.environ["TFHUB_CACHE_DIR"] = "/home/mohamed.rehan/Research/super_resolution/model_cache/"
os.environ["TFHUB_CACHE_DIR"] = "/home/mohamed.rehan/Research/super_resolution/model_cache/"


hr_shape=(384,384,3)
lr_shape=(hr_shape[0]/4,hr_shape[1]/4,hr_shape[2])

def load_images_from_folder(folder): 
    images = [] 
    for filename in os.listdir(folder): 
        img = Image.open(os.path.join(folder,filename)) 
        img_resized=np.array(img)
        images.append(img_resized) 
    return images

output= load_images_from_folder('/data/super_resolution/outputhr_processed')
input= load_images_from_folder('/data/super_resolution/input')


# Takes list of images and provide HR and LR images in form of numpy array
# Takes list of images and provide HR images in form of numpy array
def high_res_images(images):
    HR_images = np.array(images)
    return HR_images 

input_ = high_res_images(input)
output_ = high_res_images(output)


def normalize_img(input_data):
    return (input_data.astype(np.float32) - 127.5)/127.5 

def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)

input = normalize_img(input_)
output = normalize_img(output_)
X_train_lr, X_test_lr, X_train_hr, X_test_hr = train_test_split(input, output, test_size=0.2, random_state=42)

###Upsample function(changing from H * W * C to H * W * 4C then to 2H * 2W * C using pixelshuffling)
def upsample(model,filter_size,no_of_channels,strides):
    #scaling factor=2
    scale=2
    no_of_filters=no_of_channels *(scale ** 2)
    model=Conv2D(filters=no_of_filters,kernel_size=filter_size,strides=strides,padding='same')(model)
    model=UpSampling2D(size=scale)(model)
    model=PReLU()(model)
    return model



def residual_block(model, kernel_size, no_of_filters, strides):
    
    gen = model
    model = Conv2D(filters = no_of_filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = no_of_filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
        
    model = add([gen, model])
    
    return model

# Using Functional API of Keras

def generator_network(gen_input):

    gen_input = Input(shape = gen_input)
     
    model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(gen_input)
    model = PReLU(shared_axes=[1,2])(model)               # each filter only has one set of parameters
    model_part1 = model
        
  # Using 16 Residual Blocks
    for i in range(16):
        model = residual_block(model, 3, 64, 1)
        # 16 residual blocks with skip connections 

    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = add([model_part1, model])                      
  #  Element wise of model_part1 and model after 16 residual blocks
     

    for i in range(2):
        model = upsample(model, 3, 64, 1)  #no of channels=64  
     
    model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)
  
    model = Activation('tanh')(model)                  # tanh activation in last layer
    
    gen_model = Model(inputs = gen_input, outputs = model)     # specifying the input and output to the model
  
    return gen_model

generator_model= generator_network(lr_shape)
# generator_model = load_model('gen_model.h5', compile=False)


def conv_disc_block(model,filters,filter_size,strides):
    model=Conv2D(filters=filters,kernel_size=filter_size,strides=strides,padding='same')(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model=LeakyReLU(alpha=0.1)(model)
    return model


def discriminator_network(image_shape):
    disc_input=Input(shape = image_shape)
  #convolution layer(k3n64s1)
    model=Conv2D(filters = 64,kernel_size = 3,strides=1,padding='same' )(disc_input)
  #Activation-leaky relu
    model=LeakyReLU(alpha=0.1)(model)
  
  #discriminator block (k3n64s2)
    model=conv_disc_block(model,64,3,2)
  #discriminator block (k3n128s1)
    model=conv_disc_block(model,128,3,1)
  #discriminator block (k3n128s2)
    model=conv_disc_block(model,128,3,2)
  #discriminator block (k3n256s1)
    model=conv_disc_block(model,256,3,1)
  #discriminator block (k3n256s2)
    model=conv_disc_block(model,256,3,2)
  #discriminator block (k3n512s1)
    model=conv_disc_block(model,512,3,1)
  #discriminator block (k3n512s2)
    model=conv_disc_block(model,512,3,2)

  #for dense layer input should be column vector/flatten
    model=Flatten()(model)
  #dense layer with 1024 nodes
    model=Dense(1024)(model)
  #Activation-leaky relu
    model=LeakyReLU(alpha=0.1)(model)

  #dense layer with 1 nodes
    model=Dense(1)(model)
  #Activation-sigmoid
    model=Activation('sigmoid')(model)
    disc_model=Model(inputs=disc_input,outputs=model)
    return disc_model


discriminator_model = discriminator_network(hr_shape)
# discriminator_model = load_model('dis_model.h5', compile=False)


# To use mean, square we need to use keras.backend
# For content loss, compare the results which the vgg19 provides which feeding the y_true and y_pred

def content_loss(image_shape):
    def loss( y_true, y_pred):
        vgg19 = VGG19(include_top=False, weights='/home/mohamed.rehan/Research/super_resolution/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=image_shape)
        vgg19.trainable = False
        for layer in vgg19.layers:
            layer.trainable = False
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False
        return K.mean(K.square(model(y_true) - model(y_pred)))
    return loss  


def gan_network(generator_model,discriminator_model,shape):
    discriminator_model.trainable = False
    gan_input=Input(shape=shape)
    print("input")
    SR=generator_model(gan_input)
    print("SR")
    gan_output=discriminator_model(SR)
    print("gan_output")
    model=Model(inputs=gan_input,outputs=[SR,gan_output])
    return model


os.makedirs('/data/super_resolution/working/Super-Resolve', exist_ok = True)
np.random.seed(10)


def train_model(batch_size,epochs):
    no_of_batches=X_train_hr.shape[0]//batch_size
    adam = Adam(lr=0.0001 ,beta_1=0.9 ,beta_2=0.999, epsilon=1e-08 )
    generator_model.compile(loss=content_loss(hr_shape), optimizer=adam)
    discriminator_model.compile(loss='binary_crossentropy',optimizer=adam)

    gan_model=gan_network(generator_model,discriminator_model,lr_shape)
    discriminator_model.trainable = False
    gan_model.compile(loss=[content_loss(hr_shape),'binary_crossentropy'],loss_weights=[1.0,1e-3],optimizer=adam)


    for i in range(0,epochs):
        print("\nEpoch    : "+ str(i))

        for j in range(no_of_batches):

            print("Batch    : "+str(j))

            rand_nums = np.random.randint(0, X_train_hr.shape[0], size=batch_size)

            image_batch_hr = X_train_hr[rand_nums]
            image_batch_lr = X_train_lr[rand_nums]

            batch_gen_sr = generator_model.predict(image_batch_lr)
            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2      ## Here we use concept of label smoothing
            fake_data_Y = np.random.random_sample(batch_size)*0.2

            discriminator_model.trainable = True
            d_loss_real = discriminator_model.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator_model.train_on_batch(batch_gen_sr, fake_data_Y)
            d_loss = 0.5*np.add(d_loss_fake, d_loss_real)

            discriminator_model.trainable = False     
            gan_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            loss_gan = gan_model.train_on_batch(image_batch_lr, [image_batch_hr,gan_data_Y])
        
        if i % 20 == 0:
            path='/data/super_resolution/working/Super-Resolve/'
            generator_model.save(path+'gen_model'+str(i)+'.h5')
            discriminator_model.save(path+'dis_model'+str(i)+'.h5')


        print("discriminator_loss : %f" % d_loss)
        print("gan_loss :", loss_gan)
        loss_gan = str(loss_gan)

        loss_file = open( 'losses.txt' , 'a')
        loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' %(i, loss_gan, d_loss) )
        loss_file.close()

tf.config.experimental_run_functions_eagerly(True)

lr_shape = tuple(map(int, lr_shape))

train_model(1, 2000)


