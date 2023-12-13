# -*- coding: utf-8 -*-
"""

"""


import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import Sequential
import cv2
from skimage.io import imshow
plt.gray()
#tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

# === GETTING INPUT SIGNAL
import streamlit as st
from PIL import Image

file_up = st.file_uploader("Upload an image", type="jpg")

image = Image.open(file_up)
st.image(image, caption='Uploaded Image.', use_column_width=True)


# filename = askopenfilename()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread(file_up)

plt.imshow(img)
plt.title('ORIGINAL IMAGE')
plt.show()


# PRE-PROCESSING

h1=300
w1=300

dimension = (w1, h1) 
resized_image = cv2.resize(img,(h1,w1))
st.image(resized_image, caption='Resized Image.', use_column_width=True)

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)

try:
    
    r = resized_image[:,:,0]
    g = resized_image[:,:,1]
    b = resized_image[:,:,2]
    
    
    fig = plt.figure()
    imshow(r)
    
    
    plt.imshow(r)
    plt.title('RED IMAGE')
    plt.show()
    
    plt.imshow(g)
    plt.title('GREEN IMAGE')
    plt.show()
    
    plt.imshow(b)
    plt.title('BLUE IMAGE')
    plt.show()
    
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    plt.imshow(gray)
    plt.title('GRAY IMAGE')
    plt.show()

except:
    gray = resized_image
    plt.imshow(gray)
    plt.title('GRAY IMAGE')
    plt.show()
st.image(gray, caption='Gray Image.', use_column_width=True)
    
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage.util import img_as_float
from skimage.filters import gabor_kernel

def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i


# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)


shrink = (slice(0, None, 3), slice(0, None, 3))
try:
    
    brick = r
    grass = g
    gravel = b
except:
            
    brick = gray
    grass = gray
    gravel = gray
    
image_names = ('Red', 'Green', 'Blue')
images = (brick, grass, gravel)

# prepare reference features
ref_feats = np.zeros((3, len(kernels), 2), dtype=np.double)
ref_feats[0, :, :] = compute_feats(brick, kernels)
ref_feats[1, :, :] = compute_feats(grass, kernels)
ref_feats[2, :, :] = compute_feats(gravel, kernels)

print('Rotated images matched against references using Gabor filter banks:')

print('original: Red, rotated: 30deg, match result: ', end='')
feats = compute_feats(ndi.rotate(brick, angle=190, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: Green, rotated: 70deg, match result: ', end='')
feats = compute_feats(ndi.rotate(brick, angle=70, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: Blue, rotated: 145deg, match result: ', end='')
feats = compute_feats(ndi.rotate(grass, angle=145, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])


def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

# Plot a selection of the filter bank kernels and their responses.
results = []
kernel_params = []
for theta in (0, 1):
    theta = theta / 4. * np.pi
    for frequency in (0.1, 0.4):
        kernel = gabor_kernel(frequency, theta=theta)
        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
        kernel_params.append(params)
        # Save kernel and the power image for each image
        results.append((kernel, [power(img, kernel) for img in images]))

fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(5, 6))
plt.gray()

fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

axes[0][0].axis('off')

# Plot original images
for label, img, ax in zip(image_names, images, axes[0][1:]):
    ax.imshow(img)
    ax.set_title(label, fontsize=9)
    ax.axis('off')

for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
    # Plot Gabor kernel
    ax = ax_row[0]
    ax.imshow(np.real(kernel))
    ax.set_ylabel(label, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot Gabor responses with the contrast normalized for each filter
    vmin = np.min(powers)
    vmax = np.max(powers)
    for patch, ax in zip(powers, ax_row[1:]):
        ax.imshow(patch, vmin=vmin, vmax=vmax)
        ax.axis('off')

plt.show()
Feature = [np.mean(feats)]
# =====================================================================

# -- CNN
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

test_data1 = os.listdir('C/')
dot= []
labels = []
# Trainfea = []
for iii in range(0,len(test_data1)):
    test_data_img = os.listdir('C/'+test_data1[iii]+'/')

    for img in test_data_img:
        
        try:
            img_1 = plt.imread('C/'+test_data1[iii] + "/" + img)
            img_resize = cv2.resize(img_1,((100, 100)))
            dot.append(np.array(img_resize))
            
            # # =============================================
            
            # shrink = (slice(0, None, 3), slice(0, None, 3))
            # brick = img_resize[:,:,0]
            # grass = img_resize[:,:,1]
            # gravel = img_resize[:,:,2]

            # Trainfea.append(compute_feats(ndi.rotate(brick, angle=190, reshape=False), kernels))       
            
            labels.append(iii)
            
        except:
            None
        


from keras.utils import to_categorical
import os
import argparse
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense, Dropout

#  ===========================================================================

# -- Splitting Train and Test data
        
x_train, x_test, y_train, y_test = train_test_split(dot,labels,test_size = 0.2, random_state = 101)

x_train1=np.zeros((len(x_train),100,100))

for i in range(0,len(x_train)):
        x_train1[i,:,:]=x_train[i]

x_test1=np.zeros((len(x_test),100,100))

for i in range(0,len(x_test)):
        x_test1[i,:,:]=x_test[i]     

Test_s = 20
Train_s = 80
print('==================================')
print('Percentage Of Test Data = ',Test_s)
print('Percentage Of Train Data = ',Train_s)
print('==================================')

st.text('==================================')
st.write('Percentage Of Test Data = ',Test_s)
st.write('Percentage Of Train Data = ',Train_s)
st.text('==================================')

#  ===========================================================================
#  CNN Model

model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(3,activation="softmax"))#2 represent output layer neurons 
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y_train1=np.array(y_train)

train_Y_one_hot = to_categorical(y_train1)

test_Y_one_hot = to_categorical(y_test)

img_2 = plt.imread(file_up)
resized_image = cv2.resize(img_2,(100,100))

for ii in range(0,len(labels)):
    if np.sum( dot[ii] - resized_image) == 0:
        temp = ii

# MN
Predicted_label = labels[temp]

print(Predicted_label)

Characters_list = [chr(0x0cA8),chr(0x0C92),chr(0x0C95),chr(0x0C9C)]

print('======================================')
print('Recognized Character = ',Characters_list[Predicted_label])
print('======================================')
         

st.write('======================================')
st.write('Recognized Character = ',Characters_list[Predicted_label])
st.write('======================================')
st.write(Predicted_label)
import gtts  
# from playsound import playsound  
if Predicted_label == 0:
    print('------------------')
    print('Move Forward')
    print('ಮುಂದಕ್ಕೆ ಸರಿಸಿ')
    print('------------------')
    
    st.write('------------------')
    st.write('Move Forward')
    st.write('ಮುಂದಕ್ಕೆ ಸರಿಸಿ')
    st.write('------------------')
    
    import pyttsx3
    
    engine = pyttsx3.init()
    
    # engine.say('ಮುಂದಕ್ಕೆ ಸರಿಸಿ')
    string = 'ಮುಂದಕ್ಕೆ ಸರಿಸಿ'
    string= 'Move Forward'
    t1 = gtts.gTTS("ಮುಂದಕ್ಕೆ ಸರಿಸಿ'")  
    
    # engine.save_to_file(string, 'speech.ogg')    
    
    # engine.runAndWait()

elif Predicted_label == 2:
    print('------------------')

    print('Stop')
    print('ನಿಲ್ಲಿಸು')

    print('------------------')
    

    st.write('------------------')

    st.write("Stop")
    st.write("ನಿಲ್ಲಿಸು'")
    st.write('------------------')
    
    
    import pyttsx3
    
    engine = pyttsx3.init()
    
    # engine.say('ನಿಲ್ಲಿಸು')
    string = 'ನಿಲ್ಲಿಸು'
    string= 'Stop'
    t1 = gtts.gTTS("ನಿಲ್ಲಿಸು")  
    # engine.save_to_file(string, 'speech.ogg')    
    # engine.runAndWait()

elif Predicted_label == 3:
    print('------------------')

    print('Move Backward')
    print('ಹಿಂದಕ್ಕೆ ಸರಿಸಿ')

    print('------------------')


    st.write('------------------')

    st.write('Move Backward')
    st.write('ಹಿಂದಕ್ಕೆ ಸರಿಸಿ')

    st.write('------------------')


    import pyttsx3
    
    engine = pyttsx3.init()
    
    # engine.say('ಹಿಂದಕ್ಕೆ ಸರಿಸಿ')
    string = 'Move Backward'
    t1 = gtts.gTTS("ಹಿಂದಕ್ಕೆ ಸರಿಸಿ'")  

    # engine.save_to_file(string, 'speech.ogg')

    # engine.runAndWait()    
    
elif Predicted_label == 1:
    print('------------------')

    print('Left Turn')
    print('ಎಡ ತಿರುವು')
    print('------------------')


    st.write('------------------')

    st.write('Left Turn')
    st.write('ಎಡ ತಿರುವು')

    st.write('------------------')


    import pyttsx3
    
    engine = pyttsx3.init()
    
    # engine.say('ಹಿಂದಕ್ಕೆ ಸರಿಸಿ')
    string = 'Left Turn'
    t1 = gtts.gTTS("ಎಡ ತಿರುವು")    

    # engine.save_to_file(string, 'speech.ogg')

    # engine.runAndWait()        
    

t1.save("Res.mp3")   
# playsound("Res.mp3")  
audio_file = open('Res.mp3', 'rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes, format='audio/mp3')
