import numpy as np

from keras.optimizers import Adam
import cv2
import h5py
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from loss import MSE,MAE
import os
from imgprocess import image_preprocessing
from LigMSANet import LigMSANet

if __name__=='__main__':

    # Settings
    net = 'LigMSANet'
    dataset = "B"

    #loading the dataset and split the training data
    train_x = np.load('./data/ShanghaiTech/part_B/train_data/train_x.npy',allow_pickle= True)
    train_y = np.load('./data/ShanghaiTech/part_B/train_data/train_y.npy',allow_pickle= True)
    print(train_x.shape,train_y.shape)
    print(test_x.shape,test_y.shape)
    #CUDA initial
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Creating my model
    optimizer = Adam(lr=1e-4, decay=5e-3)
    model = LigMSANet(input_shape=(None, None, 3))
    model.compile(optimizer=optimizer, loss = MSE, metrics = [MAE,MSE])
    model.summary()

    #model saving
    if not os.path.exists('models'):
        os.makedirs('models')
    with open('./models/{}.json'.format(net), 'w') as fout:
        fout.write(model.to_json())

    # Settings of training
    batch_size = 8
    

    # model checkpoint
    file_path = './weights_B/weights-improvement-{epoch:03d}-{val_MAE:.4f}-{val_MSE:.4f}.h5'
    #checkpoint = ModelCheckpoint(file_path, monitor='val_MSE',
                                # verbose=1, save_best_only=True, mode='min')
    checkpoint_MAE = ModelCheckpoint(file_path, monitor='val_MAE', verbose=1, save_best_only=True, mode='min')
    checkpoint_RMSE = ModelCheckpoint(file_path, monitor='val_MSE', verbose=1, save_best_only=True, mode='min')                            
    #model earlystopping
    early_stop = EarlyStopping(monitor='val_MAE', patience=5, verbose=1)

    #monitoring the learning rate
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_MSE', patience=2,
                                                verbose=1, factor=0.5, min_lr=1e-10)

    #model fitting process
    history = model.fit(train_x,train_y,
                        batch_size=batch_size,
                        epochs=100,
                        validation_split=0.125,
                        shuffle= True,
                        verbose=1,
                        callbacks=[checkpoint_MAE, checkpoint_MSE],
                        validation_freq=1)




















