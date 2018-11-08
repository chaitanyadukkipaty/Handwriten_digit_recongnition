from keras.datasets import mnist

(xtrain,ytrain),(xtest,ytest) = mnist.load_data()

from keras.utils import to_categorical

ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)

print(ytrain.shape,ytest.shape)

xtrain = xtrain/255
xtest = xtest/255

xtrain = xtrain.reshape(xtrain.shape[0],1,28,28)
xtest = xtest.reshape(xtest.shape[0],1,28,28)

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')
#fix randomseed for reproducibility
seed=7
np.random.seed(seed)

model = Sequential()
model.add(Conv2D(30,(5,5),input_shape=(1,28,28), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(15,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='softmax'))
#Compile model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

model.fit(xtrain,ytrain,epochs=10,batch_size=200)
#Final evaluation of the model
scores = model.evaluate(xtest,ytest,verbose=0)
print("large CNN Error: ",str(1000-scores[1]*100),"%")

model.save("model.h5")
print("model saved as model.h5")

from  keras.models  import model_from_json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print("model saved as model.json file")