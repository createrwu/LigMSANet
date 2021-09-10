# coding=gbk
import os     
import numpy as np     
import random     
import time     
import codecs     
import sys     
import math
import cv2
import scipy
from keras import backend as K
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
from scipy.io import loadmat
from keras.models import model_from_json
#from wnet import wnet
import time
import keras
from layers.attention import PAM, CAM
from keras.layers import Layer
from keras.layers.core import Lambda
import tensorflow as tf
from tensorflow.keras import layers


def hard_swish(x):

    return x * K.relu(x+3, max_value=6)/6

def mse_loss(y_true,y_pred):
  se = K.sum(K.square(y_true - y_pred), axis=[1,2,3])
  mse = K.mean(se)
  return mse
   
def RMSE(y_true,y_pred):
  y_t = K.sum(y_true,axis=[1,2,3])
  y_p = K.sum(y_pred,axis=[1,2,3])
  se = K.square(y_p - y_t)
  mse = K.mean(se)
  rmse = K.sqrt(mse)
  return rmse

def MAE(y_true,y_pred):
  y_t = K.sum(y_true,axis=[1,2,3])
  y_p = K.sum(y_pred,axis=[1,2,3])
  ae = K.abs(y_p - y_t)
  mae = K.mean(ae)
  return mae


def norm_by_imagenet(img):
    if len(img.shape) == 3:
        img = img / 255.0
        img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
        img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
        return img
    elif len(img.shape) == 4 or len(img.shape) == 1:
        # In SHA, shape of images varies, so the array.shape is (N, ), that's the '== 1' case.
        imgs = []
        for im in img:
            im = im / 255.0
            im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
            im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
            im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225
            imgs.append(im)
        return np.array(imgs)
    else:
        print('Wrong shape of the input.')
        return None
    

def get_img_path(paths_test='./data/paths_train_val_test/paths_B/paths_test.txt'):
    with open(paths_test, 'r') as fin:
        img_paths = sorted(
            [l.rstrip() for l in fin.readlines()],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
    with open(paths_test, 'r') as fin:
        point_paths = sorted(
            [l.rstrip().replace('images', 'ground_truth').replace('.jpg', '.mat').replace('IMG','GT_IMG') for l in fin.readlines()],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
    return img_paths,point_paths

def img_resize(img_paths, unit=16):
    
    img = cv2.cvtColor(cv2.imread(img_paths), cv2.COLOR_BGR2RGB).astype(np.float32)
    h,w = img.shape[0], img.shape[1]
    if h < 400:
        img = cv2.copyMakeBorder(img, 0, 400-h, 0, 0, cv2.BORDER_CONSTANT, value=0)
        h = img.shape[0]
    if w < 400:
        img = cv2.copyMakeBorder(img, 0, 0, 0, 400-w, cv2.BORDER_CONSTANT, value=0)
        w = img.shape[1]
    if h%unit:
        img = cv2.copyMakeBorder(img, 0, unit-h % unit, 0, 0, cv2.BORDER_CONSTANT, value=0)
    if w%unit:
        img = cv2.copyMakeBorder(img, 0, 0, 0, unit-w % unit, cv2.BORDER_CONSTANT, value=0)
    img = norm_by_imagenet(img)
    img = img[np.newaxis,:]
    return img

def Create_Point_Matrix(point_path):

    img_path = point_path.replace('.mat', '.jpg').replace('ground_truth','images').replace('GT_IMG_','IMG_')
    pts = loadmat(point_path)
    img = cv2.imread(img_path)
    k = np.zeros((img.shape[0], img.shape[1]))
    gt = pts["image_info"][0, 0][0, 0][0]
    for i in range(len(gt)):
      if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    return k
  

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
    error_list = []
    time_list = []
    img_paths,point_paths=get_img_path()
    with open('./models/dw_3.json', 'r') as file:
      model_json = file.read()
    model = model_from_json(model_json, custom_objects={"hard_swish":hard_swish,'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D,'PAM':PAM,'CAM':CAM, 'tf': tf})
    model.load_weights('./weights_B/weights-improvement-014-1.8430-2.6278.h5',by_name=True)
    model.compile(optimizer='Adam',loss=mse_loss,metrics=[RMSE,MAE])    
    print('num：',len(img_paths))
    for i in range(len(img_paths)):
        img=img_resize(img_paths[i])
        t1 = time.time()
        predict = model.predict(img)
        #np.save('./predict/B/{}_predict.npy'.format(i+1),predict)
        t2 = time.time()
        t = t2 - t1
        predict = np.sum(predict)
        predict = np.around(predict)
        gt = loadmat(point_paths[i])
        gt = gt["image_info"][0, 0][0, 0][0]
        gt = gt.shape[0]
        error_list.append(predict - gt)
        time_list.append(t)
        print("the image id {}, the true number is {}, the prediction is {}, error is {}".format(i+1,gt,predict,predict-gt))
        
    
    mae = np.abs(error_list)
    mae = np.mean(mae)
    rms= np.power(error_list,2)
    rms = np.mean(rms)
    rms = np.sqrt(rms)
    avg_t = np.mean(time_list)
    print(mae,rms)
    print(avg_t)








