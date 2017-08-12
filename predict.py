# -*- coding: utf-8 -*-
'''
Created on Jun 14, 2014

@author: Gedas
'''
import Image
import random
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from skimage import io
import scipy.ndimage as ndimage
import scipy.io
import numpy.matlib
#import scipy.sparse as sps
from scipy.sparse import csr_matrix
from scipy.sparse import dia_matrix
import cv2

import sys


caffe_root='/path/to/caffe/'

sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0, '/usr/local/cuda-7.0/lib')
import caffe


caffe.set_mode_gpu()
caffe.set_device(0)

model_file='/path/to/caffemodel'

## RGB + DHG Model
deploy_file='EgoNet_RGB_DHG.prototxt'

## RGB Model
#deploy_file='EgoNet_RGB.prototxt'

## input/output files
img_dir='RGB_images/'
dhg_dir='DHG_images/'

output_dir='results/'
if not os.path.exists(output_dir):
   os.mkdir(output_dir)


## Loading the model
if os.path.exists(model_file):
       print 'Loading network...'
       net = caffe.Classifier(deploy_file, model_file)
else:
       print 'Network file doesnt exist!'
       print model_file
       sys.exit(1)
       

files=os.listdir(img_dir)

for ff in files:
   if '.jpg' in ff:

      ## Reading image file
      cur_img_file=img_dir+ff
      cur_im = Image.open(cur_img_file)
      cur_im = np.array(cur_im, dtype=np.float32)
              
      cur_dhg_file=dhg_dir+ff
      cur_dhg_im = Image.open(cur_dhg_file)
      cur_dhg_im = np.array(cur_dhg_im, dtype=np.float32)
      
      
      ## RGB to BGR + mean subtraction
      cur_im = cur_im[:,:,::-1]
      cur_im -= np.array((103.939, 116.779, 123.68))
      
      ## DHG to  + mean subtraction
      cur_dhg_im = cur_dhg_im[:,:,::-1]
      cur_dhg_im -= np.array((54.21, 82.37, 92.07))
      
      
      num_rows=cur_im.shape[0]
      num_cols=cur_im.shape[1]
             
      print 'Predicting...'
      
      ## Setting RGB data
      in_ = cur_im
      in_ = in_.transpose((2,0,1))
      
      net.blobs['data'].reshape(1, *in_.shape)
      net.blobs['data'].data[...] = in_
 
      ## Setting DHG data
      if 'dhg_data' in net.blobs.keys():
         dhg_in_ = cur_dhg_im
         dhg_in_ = dhg_in_.transpose((2,0,1))
      
         net.blobs['dhg_data'].reshape(1, *dhg_in_.shape)
         net.blobs['dhg_data'].data[...] = dhg_in_
     
      ## Setting XY data
      XY=np.mgrid[0:num_rows,0:num_cols]
      X=XY[1,:,:]/float(num_cols)*255
      Y=XY[0,:,:]/float(num_rows)*255
      junk=Y/float(num_rows)*255
      
      spatial_data=np.zeros((num_rows,num_cols,3))
      spatial_data[:,:,0]=X
      spatial_data[:,:,1]=Y
      spatial_data[:,:,2]=junk

      spatial_data -= np.array((127.25, 127.16, 0.00))
      spatial_data = spatial_data.transpose((2,0,1))

      net.blobs['spatial_data'].reshape(1, *spatial_data.shape)
      net.blobs['spatial_data'].data[...] = spatial_data
    
      ## Forward Pass
      net.forward()
      
      print 'Done predicting!'
                      
      ## Outputting the data
      layer_key='fc10_sm'
      fc10 = net.blobs[layer_key].data[0][:,:,:]
      fc10=np.transpose(fc10, (1, 2, 0))
      fc10=fc10[:,:,1]
      
      print 'Outputting...'
      output_path=output_dir+ff
      #scipy.io.savemat(output_path, mdict={'data': fc10})
      cv2.imwrite(output_path, fc10*255)

