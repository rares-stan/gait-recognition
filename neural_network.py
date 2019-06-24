from keras.layers import Conv3D, MaxPooling3D, Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import os
import numpy as np
import pickle


OUTPUT_H = 120
OUTPUT_W = 60
OUTPUT_FRAMES = 50

NAME = 'test'
BATCH_SIZE = 1
NUM_EPOCHS = 1000
OUTPUT_SIZE = 62


def one_hot_y(y):
    r = []
    for output in y:
        temp = [0] * OUTPUT_SIZE
        pers_index = int(output) - 1
        if pers_index >= OUTPUT_SIZE:
            pers_index = OUTPUT_SIZE
        temp[pers_index] = 1
        r.append(temp)
    return np.array(r)


f_train1 = open('data/train1.pickle', 'rb')
train_data1 = pickle.load(f_train1)
f_train2 = open('data/train2.pickle', 'rb')
train_data2 = pickle.load(f_train2)
f_train3 = open('data/train3.pickle', 'rb')
train_data3 = pickle.load(f_train3)
f_train4 = open('data/train4.pickle', 'rb')
train_data4 = pickle.load(f_train4)

f_val = open('data/val.pickle', 'rb')
val_data = pickle.load(f_val)

f_test = open('data/test.pickle', 'rb')
test_data = pickle.load(f_test)

train_data = np.concatenate([train_data1, train_data2, train_data3, train_data4])

x_train, y_train = zip(*train_data)
x_val, y_val = zip(*val_data)
x_test, y_test = zip(*test_data)

x_train = np.reshape(np.array(x_train), (248, OUTPUT_FRAMES, OUTPUT_H, OUTPUT_W, 1))
x_val = np.reshape(np.array(x_val), (62, OUTPUT_FRAMES, OUTPUT_H, OUTPUT_W, 1))
x_test = np.reshape(np.array(x_test), (62, OUTPUT_FRAMES, OUTPUT_H, OUTPUT_W, 1))
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

y_train = one_hot_y(y_train)
y_test = one_hot_y(y_test)
y_val = one_hot_y(y_val)


model = Sequential()
model.add(Conv3D(32, (10, 10, 10), padding='same', input_shape=(OUTPUT_FRAMES, OUTPUT_H, OUTPUT_W, 1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# model.add(Dropout(0.3))

model.add(Conv3D(16, (10, 10, 10), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# model.add(Dropout(0.3))

model.add(Conv3D(8, (10, 10, 10), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# model.add(Dropout(0.3))

model.add(Conv3D(4, (10, 10, 10), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(OUTPUT_SIZE, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
nume_folder = '\\' + NAME + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")

os.mkdir(os.getcwd()+'\\models'+nume_folder)
filepath = os.getcwd()+'\\models'+nume_folder+"\\weights-improvement-{epoch:02d}-{val_acc:.6f}.h5"

tensorboard = TensorBoard(log_dir='./logs'+nume_folder, histogram_freq=0, write_graph=True, write_images=False)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
earlystopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=20,  restore_best_weights=False)


history = model.fit(
    x_train, y_train, validation_data=(x_val, y_val), shuffle=True, batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS, verbose=2, callbacks=[tensorboard, checkpoint, earlystopping]
)

loss_and_metrics = model.evaluate(x_test, y_test, verbose=2)
print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])
