"""
File: denoising_cnn.py
Author: Ariel Hernan Curiale
Email: curiale@gmail.com
Description: Ejemplo para eliminar el ruido en imagenes tomado del blog oficial
de keras https://blog.keras.io/building-autoencoders-in-keras.html
"""
import os
import matplotlib.pyplot as plt
import numpy as np

# Cargamos y generamos los datos

tmp = np.load('datos/mnist_database.npz')
x_train, y_train = tmp['x_train'], tmp['y_train']
x_test, y_test = tmp['x_test'], tmp['y_test']
#nimg = 5000 # Reducimos el tamaño de la BD -> reduce la el rendimiento
#x_train = x_train[:nimg, ...]
#y_train = y_train[:nimg, ...]
#x_test = x_test[:nimg, ...]
#y_test = x_test[:nimg, ...]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
print(x_train.shape)
print(x_test.shape)

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if it is using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if it is using `channels_first` image data format

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0,
        size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Creamos el modelo
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
input_layer = Input(shape=(28, 28, 1))  # adapt this if it is using `channels_first` image data format
nfilters = 32
kernel_size = [3,3]

layer = Conv2D(nfilters, kernel_size, use_bias=True,
              activation='relu', padding='same')(input_layer)
layer = MaxPooling2D(pool_size=(2, 2), padding='same')(layer)
layer = Conv2D(nfilters, kernel_size, use_bias=True,
activation='relu', padding='same')(layer)
encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(layer)
# at this point the representation is (7, 7, 32)
layer = Conv2D(nfilters, kernel_size, use_bias=True,
activation='relu', padding='same')(encoded)
layer = UpSampling2D(size=(2, 2))(layer)
layer = Conv2D(nfilters, kernel_size, use_bias=True,
activation='relu', padding='same')(layer)
layer = UpSampling2D(size=(2, 2))(layer)
decoded = Conv2D(1, kernel_size, use_bias=True,
activation='sigmoid', padding='same')(layer)

model = Model(input_layer, decoded)
# Compilamos el modelo
model.compile(optimizer='adadelta', loss='binary_crossentropy',
              metrics=['accuracy'])
# Entrenamos la CNN
fweights = 'denoising_cnn_weights.h5'
if os.path.isfile(fweights):
    from keras.models import load_model
    model.load_weights(fweights)
else:
    history = model.fit(x_train_noisy, x_train,
                epochs=5,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test), verbose=1)

    model.save_weights(fweights)

x_test_denoisy = model.predict(x_test_noisy, verbose=1)

# Visualizamos los resultados Matplotlib
import matplotlib.pyplot as plt
f, ax = plt.subplots(1,3)
ax[0].imshow(x_test[0, ..., 0], 'gray')
ax[1].imshow(x_test_noisy[0, ..., 0], 'gray')
ax[2].imshow(x_test_denoisy[0, ..., 0], 'gray')
f.savefig('denoisingDigits.jpg', bbox_inches='tight')

if not os.path.isfile(fweights):
    plt.figure()
    plt.title('Error on training performance')
    plt.plot(history.history['loss'], '-b', label='loss')
    plt.plot(history.history['val_loss'], '-r', label='val_loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('Error loss')
    plt.legend()

