import scipy
from scipy import signal
import mxnet as mx
from matplotlib import pyplot as plt
from matplotlib import colors
from mxnet import gluon
import glob
import numpy as np
import os
from PIL import Image

# create dataloader (batch_size, 1, 100, 100)
def create_dataset(path, batch_size, shuffle):
  files = glob.glob(path)
  data = np.zeros((len(files),1,100,100))

  for idx, filename in enumerate(files):
    im = Image.open(filename)
    im = im.resize((100,100))
    data[idx,0,:,:] = np.array(im, dtype=np.float32)/255.0

  dataset = gluon.data.ArrayDataset(mx.nd.array(data, dtype=np.float32))
  dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  
  return dataloader

# create dataloader (batch_size, 10, 227, 227)
def create_dataset_stacked_images(path, batch_size, shuffle):
  
  files = glob.glob(path)

  # augment data by factor 5: create stacked images at 
  # different timesteps e.g. 0-10, 2-12, 4-14...
  x = 5 * int(len(files)/10) 
  data = np.zeros((x,10,227,227))
  counter = 0 
  for start in range(0,10,2):
    # create first stacked images from 0-10,10-20...
    # then create 2-12,12-22... 
    for idx in range(start, len(files)/10):
      if idx*10+10 <= len(files):
        for i in range(0,10):
          im = Image.open(files[idx*10+i])
          data[counter,i,:,:] = np.array(im, dtype=np.float32)/255.0
        counter = counter + 1
  
  dataset = gluon.data.ArrayDataset(mx.nd.array(data, dtype=np.float32))
  dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size, last_batch='rollover',shuffle=shuffle)
  
  return dataloader

# perform inference and plot results
def plot_images(path, output_directory, model, params_file, ctx):

  dataloader = create_dataset(path, batch_size=1, shuffle=False)
  counter = 0
  model.load_parameters(params_file, ctx=ctx)
  
  try:
    os.mkdir(output_directory)
  except:
    pass

  for image in dataloader:

    # perform inference
    image = image.as_in_context(ctx)
    reconstructed = model(image)

    #compute difference between reconstructed image and input image 
    reconstructed = reconstructed.asnumpy()
    image = image.asnumpy()
    diff = np.abs(reconstructed-image)

    # perform convolution on diff matrix
    # of neighboring pixels are anamolous, then mark the current pixel
    H = signal.convolve2d(diff[0,0,:,:], np.ones((4,4)), mode='same')
    x,y = np.where(H > 4)

    # plt input image, reconstructed image and difference between both
    fig, (ax0, ax1, ax2,ax3) = plt.subplots(ncols=4, figsize=(10, 5))
    ax0.set_axis_off()
    ax1.set_axis_off()
    ax2.set_axis_off()

    ax0.imshow(image[0,0,:,:], cmap=plt.cm.gray, interpolation='nearest')
    ax0.set_title('input image')
    ax1.imshow(reconstructed[0,0,:,:], cmap=plt.cm.gray, interpolation='nearest')
    ax1.set_title('reconstructed image')
    ax2.imshow(diff[0,0,:,:], cmap=plt.cm.viridis, vmin=0, vmax=1, interpolation='nearest')
    ax2.set_title('diff ')
    ax3.imshow(image[0,0,:,:], cmap=plt.cm.gray, interpolation='nearest')
    ax3.scatter(y,x,color='red',s=0.3)
    ax3.set_title('anomalies')
    plt.axis('off')
    
     # save figure
    counter = counter + 1
    fig.savefig(output_directory + "/" + str(counter) + '.png', bbox_inches = 'tight', pad_inches = 0.5)
    plt.close(fig)
