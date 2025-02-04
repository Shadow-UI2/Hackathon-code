import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, Flatten, Dense

class Meso4:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            Conv2D(8, (3,3), padding='same', input_shape=(256, 256, 3)),
            BatchNormalization(), Activation('relu'),
            AveragePooling2D(pool_size=(2,2), padding='same'),
            
            Conv2D(8, (5,5), padding='same'),
            BatchNormalization(), Activation('relu'),
            AveragePooling2D(pool_size=(2,2), padding='same'),
            
            Conv2D(16, (5,5), padding='same'),
            BatchNormalization(), Activation('relu'),
            AveragePooling2D(pool_size=(2,2), padding='same'),
            
            Conv2D(16, (5,5), padding='same'),
            BatchNormalization(), Activation('relu'),
            AveragePooling2D(pool_size=(4,4), padding='same'),
            
            Flatten(),
            Dense(16), Activation('relu'),
            Dense(1), Activation('sigmoid')
        ])
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def load_weights(self, path="Meso4_DF.h5"):
        self.model.load_weights(path)

    def predict(self, image):
        import numpy as np
        return self.model.predict(image,verbose=0)
