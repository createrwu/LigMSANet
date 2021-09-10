import numpy as np

from keras.optimizers import Adam
import cv2
import h5py
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from loss import MSE,BCE,RMSE,MAE,ssim_eucli_loss,ssim_loss,bayesian_loss
import os
#from DataHandle import gen_var_from_paths,gen_density_map_gaussian
from imgprocess import image_preprocessing
#from MFSAM import MFSAM 
from w_dw_3 import dw_3
def ssim_loss(y_true, y_pred, c1=0.01**2, c2=0.03**2):

  weights_initial = np.multiply(cv2.getGaussianKernel(11, 1.5),cv2.getGaussianKernel(11, 1.5).T)

  weights_initial = weights_initial.reshape(*weights_initial.shape, 1, 1)
  weights_initial = K.cast(weights_initial, tf.float32)
  
  mu_F = tf.nn.conv2d(y_pred, weights_initial, [1, 1, 1, 1], padding='SAME')
  mu_Y = tf.nn.conv2d(y_true, weights_initial, [1, 1, 1, 1], padding='SAME')
  mu_F_mu_Y = tf.multiply(mu_F, mu_Y)
  mu_F_squared = tf.multiply(mu_F, mu_F)
  mu_Y_squared = tf.multiply(mu_Y, mu_Y)
  sigma_F_squared = tf.nn.conv2d(tf.multiply(y_pred, y_pred), weights_initial, [1, 1, 1, 1], padding='SAME') - mu_F_squared
  sigma_Y_squared = tf.nn.conv2d(tf.multiply(y_true, y_true), weights_initial, [1, 1, 1, 1], padding='SAME') - mu_Y_squared
  sigma_F_Y = tf.nn.conv2d(tf.multiply(y_pred, y_true), weights_initial, [1, 1, 1, 1], padding='SAME') - mu_F_mu_Y
  
  ssim = ((2 * mu_F_mu_Y + c1) * (2 * sigma_F_Y + c2)) / ((mu_F_squared + mu_Y_squared + c1) * (sigma_F_squared + sigma_Y_squared + c2))
  ssim = 1 - tf.reduce_mean(ssim, reduction_indices=[1,2,3])
  ssim = tf.reduce_mean(ssim)
  return ssim

if __name__=='__main__':

    # Settings
    net = 'dw_3'
    dataset = "B"

    #loading the dataset and split the training data
    train_x = np.load('./data/ShanghaiTech/part_B/train_data/train_50_x.npy',allow_pickle= True)
    train_y = np.load('./data/ShanghaiTech/part_B/train_data/train_50_y.npy',allow_pickle= True)
    test_x = np.load('./data/ShanghaiTech/part_B/train_data/test_x.npy',allow_pickle= True)
    test_y = np.load('./data/ShanghaiTech/part_B/train_data/test_y.npy',allow_pickle= True)
    print(train_x.shape,train_y.shape)
    print(test_x.shape,test_y.shape)
    #CUDA initial
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Creating my model
    optimizer = Adam(lr=1e-4, decay=5e-3)
    model = dw_3(input_shape=(None, None, 3))
    model.compile(optimizer=optimizer, loss = MSE, metrics = [MAE,RMSE])
    model.summary()

    #model saving
    if not os.path.exists('models'):
        os.makedirs('models')
    with open('./models/{}.json'.format(net), 'w') as fout:
        fout.write(model.to_json())

    # Settings of training
    batch_size = 8
    

    # model checkpoint
    file_path = './weights_B/weights-improvement-{epoch:03d}-{val_MAE:.4f}-{val_RMSE:.4f}.h5'
    #checkpoint = ModelCheckpoint(file_path, monitor='val_RMSE',
                                # verbose=1, save_best_only=True, mode='min')
    checkpoint_MAE = ModelCheckpoint(file_path, monitor='val_MAE', verbose=1, save_best_only=True, mode='min')
    checkpoint_RMSE = ModelCheckpoint(file_path, monitor='val_RMSE', verbose=1, save_best_only=True, mode='min')                            
    #model earlystopping
    early_stop = EarlyStopping(monitor='val_MAE', patience=5, verbose=1)

    #monitoring the learning rate
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_RMSE', patience=2,
                                                verbose=1, factor=0.5, min_lr=1e-10)

    #model fitting process
    history = model.fit(train_x,train_y,
                        batch_size=batch_size,
                        epochs=100,
                        #validation_split=0.125,
                        validation_data=[test_x,test_y],
                        shuffle= True,
                        verbose=1,
                        callbacks=[checkpoint_MAE, checkpoint_RMSE],
                        validation_freq=1)




















