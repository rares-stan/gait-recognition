from keras.layers import Conv3D, MaxPooling3D, Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import os
import numpy as np
import pickle


OUTPUT_H = 120
OUTPUT_W = 60
OUTPUT_FRAMES = 30

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


f_train1 = open('data/train1-30-frames.pickle', 'rb')
train_data1 = pickle.load(f_train1)
f_train2 = open('data/train2-30-frames.pickle', 'rb')
train_data2 = pickle.load(f_train2)
f_train3 = open('data/train3-30-frames.pickle', 'rb')
train_data3 = pickle.load(f_train3)
f_train4 = open('data/train4-30-frames.pickle', 'rb')
train_data4 = pickle.load(f_train4)

f_val = open('data/validation-30-frames.pickle', 'rb')
val_data = pickle.load(f_val)

f_test = open('data/test-30-frames.pickle', 'rb')
test_data = pickle.load(f_test)

train_data = np.concatenate([train_data1, train_data2, train_data3, train_data4])

x_train, y_train = zip(*train_data)
x_val, y_val = zip(*val_data)
x_test, y_test = zip(*test_data)

x_train = np.reshape(np.array(x_train), (len(x_train), OUTPUT_FRAMES, OUTPUT_H, OUTPUT_W, 1))
x_val = np.reshape(np.array(x_val), (len(x_val), OUTPUT_FRAMES, OUTPUT_H, OUTPUT_W, 1))
x_test = np.reshape(np.array(x_test), (len(x_test), OUTPUT_FRAMES, OUTPUT_H, OUTPUT_W, 1))
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

y_train = one_hot_y(y_train)
y_test = one_hot_y(y_test)
y_val = one_hot_y(y_val)


def create_new_model():
    new_model = Sequential()
    new_model.add(Conv3D(32, (10, 10, 10), padding='same', input_shape=(OUTPUT_FRAMES, OUTPUT_H, OUTPUT_W, 1)))
    new_model.add(Activation('elu'))
    new_model.add(BatchNormalization())
    new_model.add(MaxPooling3D(pool_size=(2, 2, 2), padding="same"))
    new_model.add(Dropout(0.5))

    new_model.add(Conv3D(16, (10, 10, 10), padding='same'))
    new_model.add(Activation('elu'))
    new_model.add(BatchNormalization())
    new_model.add(MaxPooling3D(pool_size=(2, 2, 2), padding="same"))
    new_model.add(Dropout(0.5))

    new_model.add(Conv3D(8, (5, 10, 10), padding='same'))
    new_model.add(Activation('elu'))
    new_model.add(BatchNormalization())
    new_model.add(MaxPooling3D(pool_size=(2, 2, 2), padding="same"))
    new_model.add(Dropout(0.5))

    new_model.add(Conv3D(4, (3, 10, 5), padding='same'))
    new_model.add(Activation('elu'))
    new_model.add(BatchNormalization())
    new_model.add(MaxPooling3D(pool_size=(2, 2, 2), padding="same"))
    new_model.add(Dropout(0.5))

    new_model.add(Flatten())
    new_model.add(Dropout(0.5))
    new_model.add(Dense(200, activation='elu'))
    new_model.add(Dense(OUTPUT_SIZE, activation='softmax'))

    new_model.summary()
    new_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return new_model


def use_existing_model(path):
    existing_model = load_model(path)
    return existing_model


model = create_new_model()
# model = use_existing_model('./models_to_be_kept/continue_training/weights-improvement-12-0.631068.h5')

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
