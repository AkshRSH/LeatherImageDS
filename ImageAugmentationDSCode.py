#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Import Libraries
from __future__ import print_function
from __future__ import absolute_import, division, print_function, unicode_literals
import PIL.Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import save_img
import matplotlib.pyplot as plt
import tensorflow as tf
config =  tf.compat.v1.ConfigProto( device_count = {'GPU': 1} ) 
sess = tf.compat.v1.Session(config=config) 
import numpy as np
from skimage.io import imread
from skimage import exposure, color
from skimage.transform import resize
import keras
tf.compat.v1.keras.backend.set_session(sess)
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# Save Transformed Image
def imshow(img, transform,cnu,flh,nu=0):
    img = PIL.Image.open(img)
    if(nu==0):
        img = transform(img)
    else:
        img=transform
    img_array = img_to_array(img)
    save_img('/Project Files/MultiClassification/LeatherDS/BDQ/Gen/'+flh+'_'+str(cnu)+'.jpg', img_array)
    print('Image'+flh+':'+str(cnu))

# Generate 9 Rotated Images, increment angle of rotation by 10 degrees each
def imgrGen(g,vl,img, zca=False, rotation=0., w_shift=0., h_shift=0., shear=0., zoom=0., h_flip=False, v_flip=False,  preprocess_fcn=None, batch_size=9):
    datagen = ImageDataGenerator(
            zca_whitening=zca,
            rotation_range=rotation,
            width_shift_range=w_shift,
            height_shift_range=h_shift,
            shear_range=shear,
            zoom_range=zoom,
            fill_mode='nearest',
            horizontal_flip=h_flip,
            vertical_flip=v_flip,
            preprocessing_function=preprocess_fcn,
            data_format=keras.backend.image_data_format())
    
    datagen.fit(img)

    i=0
    for img_batch in datagen.flow(img, batch_size=9, shuffle=False):
        for img in img_batch:            
            img_array = img_to_array(img)
            save_img('/Project Files/MultiClassification/LeatherDS/BDQ/Gen/'+str(g)+'_'+str(vl)+'.jpg', img_array)
            print("IMG"+str(g)+'-'+str(vl))
            vl=vl+1 
            i=i+1
        if i >= batch_size:
            break

# The Image Generation Loop runs for all i = 1 to 355. i denotes the number of the original image.
for i in range (1,355):
        ct=1  #Augmented Image Number for the i th original image
        flst=str(i)
        imgflc='/Project Files/MultiClassification/LeatherDS/BDQ/'+flst+'.jpg' #Path from where the original image is picked up
        loader_transform = transforms.Resize((2506,2773))  # Transform Image with Resize Operation with the Dimensions 2506, 2773
        imshow(imgflc,loader_transform,ct,flst) # Save Image with the above transform 
        ct=ct+1
        loader_transform = transforms.Resize((1253,1386)) # Transform Image with Resize Operation with the Dimensions 1253, 1386
        imshow(imgflc,loader_transform,ct,flst)
        ct=ct+1
        loader_transform = transforms.CenterCrop(1000) #  Transform Image with Center Crop Operation 
        imshow(imgflc,loader_transform,ct,flst)
        ct=ct+1
        loader_transform = transforms.CenterCrop(800)  # Transform Image with Center Crop Operation
        imshow(imgflc,loader_transform,ct,flst)
        ct=ct+1
        loader_transform = transforms.CenterCrop(500)  # Transform Image with Center Crop Operation
        imshow(imgflc,loader_transform,ct,flst)
        ct=ct+1
        loader_transform = transforms.CenterCrop(400)  # Transform Image with Center Crop Operation
        imshow(imgflc,loader_transform,ct,flst)
        ct=ct+1
        loader_transform = transforms.RandomHorizontalFlip(p=1) # Transform Horizontal Flip
        imshow(imgflc,loader_transform,ct,flst)
        ct=ct+1
        loader_transform = transforms.RandomVerticalFlip(p=1) # Transform Vertical Flip
        imshow(imgflc,loader_transform,ct,flst)
        ct=ct+1
        imgop=PIL.Image.open(imgflc)
        loader_transform = transforms.functional.adjust_contrast(imgop,1.5) # Transform Contrast Modify 1.5
        imshow(imgflc,loader_transform,ct,flst,1)
        ct=ct+1
        loader_transform = transforms.functional.adjust_contrast(imgop,2) # Transform Contrast Modify 2
        imshow(imgflc,loader_transform,ct,flst,1)
        ct=ct+1
        loader_transform= transforms.functional.adjust_brightness(imgop,1.2) # Transform Brightness Modify 1.2
        imshow(imgflc,loader_transform,ct,flst,1)
        ct=ct+1
        loader_transform= transforms.functional.adjust_brightness(imgop,1.6) # Transform Brightness Modify 1.6
        imshow(imgflc,loader_transform,ct,flst,1)
        ct=ct+1
        loader_transform = transforms.functional.adjust_saturation(imgop,1.3)# Transform Saturation Modify 1.3
        imshow(imgflc,loader_transform,ct,flst,1)
        ct=ct+1
        loader_transform = transforms.functional.adjust_saturation(imgop,1.6) #Transform Saturation Modify 1.6
        imshow(imgflc,loader_transform,ct,flst,1)
        ct=ct+1
        #Transform Operation Modify Loop using HUE 
        hu=-4
        ct=23
        j=0.3
        loader_transform = transforms.functional.adjust_hue(imgop,j)
        imshow(imgflc,loader_transform,ct,flst,1)
        for j in range (1,14):
            j=hu/10
            loader_transform = transforms.functional.adjust_hue(imgop,j)
            imshow(imgflc,loader_transform,ct,flst,1)
            ct=ct+1
            hu=hu+0.5
        # Transform Grayscale 
        loader_transform = transforms.Grayscale()
        imshow(imgflc,loader_transform,ct,flst)
        ct=ct+1
        # Transform Rotation Operation 
        img=imread(imgflc)
        img = img.astype('float32')
        img /= 255
        h_dim = np.shape(img)[0]
        w_dim = np.shape(img)[1]
        num_channel = np.shape(img)[2]
        img = img.reshape(1, h_dim, w_dim, num_channel)
        imgrGen(i,29,img, rotation=10, h_shift=0)

