# -*- coding: utf-8 -*-
"""oxford_pet_segmentation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_99wCif8lTke9F8KMZmWRMnYlVvJHdIK
"""

## library import
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import re
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

## google drive에서 압축된 dataset download
import gdown
url = 'https://drive.google.com/uc?id=1dIR9ANjUsV9dWa0pS9J0c2KUGMfpIRG0'
fname = 'oxford_pet.zip'
gdown.download(url, fname, quiet=False)

## 압축풀기
!unzip -q oxford_pet.zip -d oxford_pet

## directory 설정
cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, 'oxford_pet')
image_dir = os.path.join(data_dir, 'images')
seg_dir=os.path.join(data_dir,'annotations','trimaps')

## image file 수 확인
image_files = [fname for fname in os.listdir(image_dir) if os.path.splitext(fname)[-1] == '.jpg']
print(len(image_files))

seg_files=[fname for fname in os.listdir(seg_dir) if os.path.splitext(fname)[-1] == '.png']
print(len(seg_files))

## image file들을 읽어서 channel이 3이 아닌 image는 삭제, seg_file도
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    seg_file=os.path.splitext(image_file)[0]+'.png'
    seg_path=os.path.join(seg_dir,seg_file)
    image = Image.open(image_path)
    image_mode = image.mode
    if image_mode != 'RGB':
        print(image_file, image_mode)
        image = np.asarray(image)
        print(image.shape)
        os.remove(image_path)
        os.remove(seg_path)

## image file 수 확인
image_files = [fname for fname in os.listdir(image_dir) if os.path.splitext(fname)[-1] == '.jpg']
print(len(image_files))

seg_files=[fname for fname in os.listdir(seg_dir) if os.path.splitext(fname)[-1] == '.png']
print(len(seg_files))

class_list = set()
for image_file in image_files:
    file_name = os.path.splitext(image_file)[0]
    class_name = re.sub('_\d+', '', file_name)
    class_list.add(class_name)
class_list = list(class_list)
print(len(class_list))

class_list.sort()
class_list

class2idx = {cls:idx for idx, cls in enumerate(class_list)}
class2idx

## train, validation directory 생성
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validation')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

image_files.sort()

cnt = 0
previous_class = ""
for image_file in image_files:
    file_name = os.path.splitext(image_file)[0]
    class_name = re.sub('_\d+', '', file_name)
    if class_name == previous_class:
        cnt += 1
    else:
        cnt = 1
    if cnt <= 160:
        cpath = train_dir
    else:
        cpath = val_dir
    image_path = os.path.join(image_dir, image_file)
    shutil.copy(image_path, cpath)
    previous_class = class_name

train_images = os.listdir(train_dir)
val_images = os.listdir(val_dir)

print(len(train_images), len(val_images))

# 임의의 이미지를 가져와서 seg map 확인
fnames=os.listdir(val_dir)
rnd_idx=random.randint(0,len(fnames)-1)
fname=fnames[rnd_idx]
fpath=os.path.join(val_dir,fname)
img=Image.open(fpath)
img=np.array(img)

# segmentation label
# 원래 label은 1: foreground, 2: background, 3: not classified 로 구성됨
# 이것을 0: background, 1: foreground & not classified 로 변경
sname=os.path.splitext(fname)[0]+'.png'
spath=os.path.join(seg_dir,sname)
seg=Image.open(spath)
seg=np.array(seg)
seg[seg>2]=1
seg[seg==2]=0

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(seg)
plt.show()

IMG_SIZE = 224
N_TRAIN=len(train_images)
N_VAL=len(val_images)

## TFRecord 저장할 directory와 file 경로 설정
tfr_dir = os.path.join(data_dir, 'tfrecord')
os.makedirs(tfr_dir, exist_ok=True)

tfr_train_dir = os.path.join(tfr_dir, 'seg_train.tfr')
tfr_val_dir = os.path.join(tfr_dir, 'seg_val.tfr')

## TFRecord writer 생성
writer_train = tf.io.TFRecordWriter(tfr_train_dir)
writer_val = tf.io.TFRecordWriter(tfr_val_dir)

# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

## Training data로 tfrecord 만들기
n_train = 0

train_files = os.listdir(train_dir)
for train_file in train_files:
    train_path = os.path.join(train_dir, train_file)
    image = Image.open(train_path)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    bimage = image.tobytes()

    file_name = os.path.splitext(train_file)[0] #Bangal_101
    class_name = re.sub('_\d+', '', file_name)
    class_num = class2idx[class_name]

    if file_name[0].islower():
      bi_cls_num=0
    else:
      bi_cls_num=1
    
    seg_name=file_name+'.png'
    seg_path=os.path.join(seg_dir,seg_name)
    seg=Image.open(seg_path)
    seg=seg.resize((IMG_SIZE,IMG_SIZE))
    seg=np.array(seg)
    seg[seg>2]=1
    seg[seg==2]=0
    bseg=seg.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
      'image': _bytes_feature(bimage),
      'cls_num': _int64_feature(class_num),
      'bi_cls_num':_int64_feature(class_num),
      'seg':_bytes_feature(bseg)
    }))
    writer_train.write(example.SerializeToString())
    n_train += 1

writer_train.close()
print(n_train)

n_val = 0

val_files = os.listdir(val_dir)
for val_file in val_files:
    val_path = os.path.join(val_dir, val_file)
    image = Image.open(val_path)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    bimage = image.tobytes()

    file_name = os.path.splitext(val_file)[0] #Bangal_101
    class_name = re.sub('_\d+', '', file_name)
    class_num = class2idx[class_name]

    if file_name[0].islower():
      bi_cls_num=0
    else:
      bi_cls_num=1
    
    seg_name=file_name+'.png'
    seg_path=os.path.join(seg_dir,seg_name)
    seg=Image.open(seg_path)
    seg=seg.resize((IMG_SIZE,IMG_SIZE))
    seg=np.array(seg)
    seg[seg>2]=1
    seg[seg==2]=0
    bseg=seg.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
      'image': _bytes_feature(bimage),
      'cls_num': _int64_feature(class_num),
      'bi_cls_num':_int64_feature(class_num),
      'seg':_bytes_feature(bseg)
    }))
    writer_val.write(example.SerializeToString())
    n_val += 1

writer_val.close()
print(n_val)

## Hyper Parameters
N_CLASS = len(class_list)
N_EPOCHS = 20
N_BATCH = 40
IMG_SIZE = 224
learning_rate = 0.0001
steps_per_epoch = N_TRAIN / N_BATCH
validation_steps = int(np.ceil(N_VAL / N_BATCH))

## tfrecord file을 data로 parsing해주는 function
def _parse_function(tfrecord_serialized):
    features={'image': tf.io.FixedLenFeature([], tf.string),
              'cls_num': tf.io.FixedLenFeature([], tf.int64),
              'bi_cls_num':tf.io.FixedLenFeature([],tf.int64),
              'seg':tf.io.FixedLenFeature([],tf.string)             
             }
    parsed_features = tf.io.parse_single_example(tfrecord_serialized, features)
    
    image = tf.io.decode_raw(parsed_features['image'], tf.uint8)    
    image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 3])
    image = tf.cast(image, tf.float32)/255. 

    cls_label = tf.cast(parsed_features['cls_num'], tf.int64)
    bi_cls_label=tf.cast(parsed_features['bi_cls_num'],tf.int64)

    seg=tf.io.decode_raw(parsed_features['seg'],tf.uint8)
    seg=tf.reshape(seg,[IMG_SIZE,IMG_SIZE,-1])
    seg=tf.cast(seg,tf.float32)
    
    return image, seg

## train dataset 만들기
train_dataset = tf.data.TFRecordDataset(tfr_train_dir)
train_dataset = train_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(N_TRAIN).prefetch(
    tf.data.experimental.AUTOTUNE).batch(N_BATCH).repeat()

## validation dataset 만들기
val_dataset = tf.data.TFRecordDataset(tfr_val_dir)
val_dataset = val_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(N_BATCH).repeat()

"""UNet like model을 random initialization으로 학습하기"""

from tensorflow.keras.layers import Input,Conv2D,Conv2DTranspose,ReLU,MaxPooling2D,Dense,BatchNormalization,GlobalAveragePooling2D,Concatenate

def create_model():
  inputs=Input(shape=(IMG_SIZE,IMG_SIZE,3))

  conv1_1=Conv2D(64,3,1,'SAME',activation='relu')(inputs)
  conv1_2=Conv2D(64,3,1,'SAME',activation='relu')(conv1_1)
  pool1_3=MaxPooling2D()(conv1_2)

  conv2_1=Conv2D(128,3,1,'SAME',activation='relu')(pool1_3)
  conv2_2=Conv2D(128,3,1,'SAME',activation='relu')(conv2_1)
  pool2_3=MaxPooling2D()(conv2_2)

  conv3_1=Conv2D(256,3,1,'SAME',activation='relu')(pool2_3)
  conv3_2=Conv2D(256,3,1,'SAME',activation='relu')(conv3_1)
  conv3_3=Conv2D(256,3,1,'SAME',activation='relu')(conv3_2)
  pool3_4=MaxPooling2D()(conv3_3)

  conv4_1=Conv2D(512,3,1,'SAME',activation='relu')(pool3_4)
  conv4_2=Conv2D(512,3,1,'SAME',activation='relu')(conv4_1)
  conv4_3=Conv2D(512,3,1,'SAME',activation='relu')(conv4_2)
  pool4_4=MaxPooling2D()(conv4_3)

  conv5_1=Conv2D(512,3,1,'SAME',activation='relu')(pool4_4)
  conv5_2=Conv2D(512,3,1,'SAME',activation='relu')(conv5_1)
  conv5_3=Conv2D(512,3,1,'SAME',activation='relu')(conv5_2)
  pool5_4=MaxPooling2D()(conv5_3)

  upconv1=Conv2DTranspose(512,5,2,'SAME',activation='relu')(pool5_4)
  concat1=Concatenate()([conv5_3,upconv1])
  conv6=Conv2D(512,3,1,'SAME',activation='relu')(concat1)

  upconv2=Conv2DTranspose(512,5,2,'SAME',activation='relu')(conv6)
  concat2=Concatenate()([conv4_3,upconv2])
  conv7=Conv2D(512,3,1,'SAME',activation='relu')(concat2)

  upconv3=Conv2DTranspose(256,5,2,'SAME',activation='relu')(conv7)
  concat3=Concatenate()([conv3_3,upconv3])
  conv8=Conv2D(256,3,1,'SAME',activation='relu')(concat3)

  upconv4=Conv2DTranspose(128,5,2,'SAME',activation='relu')(conv8)
  concat4=Concatenate()([conv2_2,upconv4])
  conv9=Conv2D(128,3,1,'SAME',activation='relu')(concat4)

  upconv5=Conv2DTranspose(64,5,2,'SAME',activation='relu')(conv9)
  concat5=Concatenate()([conv1_2,upconv5])
  conv10=Conv2D(64,3,1,'SAME',activation='relu')(concat5)

  conv11=Conv2D(64,3,1,'SAME',activation='relu')(conv10)

  conv12=Conv2D(2,1,1,'SAME',activation='softmax')(conv11)

  return keras.Model(inputs=inputs,outputs=conv12)

model=create_model()
model.summary()

lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                          decay_steps=steps_per_epoch*5,
                                                          decay_rate=0.4,
                                                          staircase=True)
model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_dataset,
    epochs=N_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps
)

## Pred image 10장 확인
idx = 0
num_imgs=10
for image,seg in val_dataset.take(num_imgs):
  plt.figure(figsize=(17,6*num_imgs))
  plt.subplot(num_imgs,3,idx*3+1)
  plt.imshow(image[0])
  plt.subplot(num_imgs,3,idx*3+2)
  plt.imshow(seg[0,:,:,0],vmin=0,vmax=1)

  plt.subplot(num_imgs,3,idx*3+3)
  prediction=model.predict(image)
  pred=np.zeros_like(prediction)

  # 0.5 이상은  1로 나머지는 0으로 변환
  thr=0.5
  pred[prediction>=thr]=1
  pred[prediction<thr]=0
  plt.imshow(pred[0,:,:,1])
  plt.show()
  idx+=1

# IOU 계산
avg_iou=0
n=0
for images, labels in val_dataset.take(validation_steps):
  preds=model.predict(images)
  preds[preds>=0.5]=1
  preds[preds<0.5]=0

  psum=labels[...,0]+preds[...,1]

  union=np.array(psum)
  union[union>1]=1.
  union=np.sum(union,axis=1)
  union=np.sum(union,axis=1)

  inter=np.array(psum)
  inter[inter==1]=0.
  inter[inter>1]=1.
  inter=np.sum(inter,axis=1)
  inter=np.sum(inter,axis=1)

  iou=inter / union
  avg_iou+=np.sum(iou)/N_VAL

print(avg_iou)

"""U-Net like model에 pretrained VGG를 활용하여 학습하기"""

from tensorflow.keras.utils import get_file
weight_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                      'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

from tensorflow.keras.utils import get_file
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
weights_path2 = keras.utils.get_file(
    'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
    WEIGHTS_PATH_NO_TOP,
    cache_subdir='models',
    md5_hash='a268eb855778b3df3c7506639542a6af')

def new_model():
  inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
  conv1_1 = Conv2D(64, 3, 1, 'SAME', activation='relu')(inputs)
  conv1_2 = Conv2D(64, 3, 1, 'SAME', activation='relu')(conv1_1)    
  pool1_3 = MaxPooling2D()(conv1_2)
    
  conv2_1 = Conv2D(128, 3, 1, 'SAME', activation='relu')(pool1_3)
  conv2_2 = Conv2D(128, 3, 1, 'SAME', activation='relu')(conv2_1)
  pool2_3 = MaxPooling2D()(conv2_2)
    
  conv3_1 = Conv2D(256, 3, 1, 'SAME', activation='relu')(pool2_3)
  conv3_2 = Conv2D(256, 3, 1, 'SAME', activation='relu')(conv3_1)
  conv3_3 = Conv2D(256, 3, 1, 'SAME', activation='relu')(conv3_2)
  pool3_4 = MaxPooling2D()(conv3_3)
    
  conv4_1 = Conv2D(512, 3, 1, 'SAME', activation='relu')(pool3_4)
  conv4_2 = Conv2D(512, 3, 1, 'SAME', activation='relu')(conv4_1)
  conv4_3 = Conv2D(512, 3, 1, 'SAME', activation='relu')(conv4_2)
  pool4_4 = MaxPooling2D()(conv4_3)
    
  conv5_1 = Conv2D(512, 3, 1, 'SAME', activation='relu')(pool4_4)
  conv5_2 = Conv2D(512, 3, 1, 'SAME', activation='relu')(conv5_1)
  conv5_3 = Conv2D(512, 3, 1, 'SAME', activation='relu')(conv5_2)
  pool5_4 = MaxPooling2D()(conv5_3)
    
  ## loading vgg16 pretrained weights
  vgg = keras.Model(inputs, pool5_4)
  vgg.load_weights(weight_path)

  upconv6 = Conv2DTranspose(512, 5, 2, 'SAME', activation='relu')(pool5_4)
  concat6 = Concatenate()([conv5_3, upconv6])
  conv6 = Conv2D(512, 3, 1, 'SAME', activation='relu')(concat6)
                              
  upconv7 = Conv2DTranspose(512, 5, 2, 'SAME', activation='relu')(conv6)
  concat7 = Concatenate()([conv4_3, upconv7])
  conv7 = Conv2D(512, 3, 1, 'SAME', activation='relu')(concat7)
    
  upconv8 = Conv2DTranspose(256, 5, 2, 'SAME', activation='relu')(conv7)
  concat8 = Concatenate()([conv3_3, upconv8])
  conv8 = Conv2D(256, 3, 1, 'SAME', activation='relu')(concat8)
    
  upconv9 = Conv2DTranspose(128, 5, 2, 'SAME', activation='relu')(conv8)
  concat9 = Concatenate()([conv2_2, upconv9])
  conv9 = Conv2D(128, 3, 1, 'SAME', activation='relu')(concat9)
    
  upconv10 = Conv2DTranspose(64, 5, 2, 'SAME', activation='relu')(conv9)
  concat10 = Concatenate()([conv1_2, upconv10])
  conv10 = Conv2D(64, 3, 1, 'SAME', activation='relu')(concat10)
    
  conv11 = Conv2D(64, 3, 1, 'SAME', activation='relu')(conv10)
    
  conv12 = Conv2D(2, 1, 1, 'SAME', activation='softmax')(conv11)
    
  return keras.Model(inputs=inputs, outputs=conv12)

new_model=new_model()
new_model.summary()

lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                          decay_steps=steps_per_epoch*5,
                                                          decay_rate=0.4,
                                                          staircase=True)
new_model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

history2 = new_model.fit(
    train_dataset,
    epochs=N_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps
)

## Pred image 10장 확인
idx = 0
num_imgs=10
for image,seg in val_dataset.take(num_imgs):
  plt.figure(figsize=(17,6*num_imgs))
  plt.subplot(num_imgs,3,idx*3+1)
  plt.imshow(image[0])
  plt.subplot(num_imgs,3,idx*3+2)
  plt.imshow(seg[0,:,:,0],vmin=0,vmax=1)

  plt.subplot(num_imgs,3,idx*3+3)
  prediction=new_model.predict(image)
  pred=np.zeros_like(prediction)

  # 0.5 이상은  1로 나머지는 0으로 변환
  thr=0.5
  pred[prediction>=thr]=1
  pred[prediction<thr]=0
  plt.imshow(pred[0,:,:,1])
  plt.show()
  idx+=1

# IOU 계산
avg_iou=0
n=0
for images, labels in val_dataset.take(validation_steps):
  preds=new_model.predict(images)
  preds[preds>=0.5]=1
  preds[preds<0.5]=0

  psum=labels[...,0]+preds[...,1]

  union=np.array(psum)
  union[union>1]=1.
  union=np.sum(union,axis=1)
  union=np.sum(union,axis=1)

  inter=np.array(psum)
  inter[inter==1]=0.
  inter[inter>1]=1.
  inter=np.sum(inter,axis=1)
  inter=np.sum(inter,axis=1)

  iou=inter / union
  avg_iou+=np.sum(iou)/N_VAL

print(avg_iou)
