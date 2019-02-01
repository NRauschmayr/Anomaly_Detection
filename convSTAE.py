import mxnet as mx
from mxnet import gluon
import numpy as np 
from mxnet.gluon import nn
import utils

# convolutional spatio-temporal autoencoder 
class ConvSTAE(gluon.nn.HybridBlock):
    def __init__(self):
        super(ConvSTAE, self).__init__()
        with self.name_scope():
            self.encoder = gluon.nn.HybridSequential(prefix="encoder")
            with self.encoder.name_scope():
                self.encoder.add(gluon.nn.Conv2D(512, kernel_size=15, strides=4, padding=0, activation='relu'))
                self.encoder.add(gluon.nn.BatchNorm())
                self.encoder.add(gluon.nn.MaxPool2D(2))
                self.encoder.add(gluon.nn.BatchNorm())
                self.encoder.add(gluon.nn.Conv2D(256, kernel_size=4, padding=0, activation='relu'))
                self.encoder.add(gluon.nn.BatchNorm())
                self.encoder.add(gluon.nn.MaxPool2D(2))
                self.encoder.add(gluon.nn.BatchNorm())
                self.encoder.add(gluon.nn.Conv2D(128, kernel_size=3, padding=0,activation='relu'))
                self.encoder.add(gluon.nn.BatchNorm())
            self.decoder = gluon.nn.HybridSequential(prefix="decoder")
            with self.decoder.name_scope():
                self.decoder.add(gluon.nn.Conv2DTranspose(channels=256, kernel_size=3, padding=0,activation='relu'))
                self.decoder.add(gluon.nn.BatchNorm())
                self.decoder.add(gluon.nn.HybridLambda(lambda F, x: F.UpSampling(x, scale=2, sample_type='nearest')))
                self.decoder.add(gluon.nn.BatchNorm())
                self.decoder.add(gluon.nn.Conv2DTranspose(channels=512, kernel_size=4, padding=0, activation='relu'))
                self.decoder.add(gluon.nn.BatchNorm())
                self.decoder.add(gluon.nn.HybridLambda(lambda F, x: F.UpSampling(x, scale=2, sample_type='nearest')))
                self.decoder.add(gluon.nn.BatchNorm())
                self.decoder.add(gluon.nn.Conv2DTranspose(channels=10, kernel_size=15, padding=0, strides=4, activation='sigmoid'))


    def hybrid_forward(self, F, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

# Train the autoencoder
def train(batch_size, ctx, num_epochs, path, lr=1e-4, wd=1e-5, params_file="autoencoder_ucsd_convstae.params"):

  # Dataloader for training dataset
  dataloader = utils.create_dataset_stacked_images(path, batch_size, shuffle=True, augment=True)

  # Get model
  model = ConvSTAE()
  model.hybridize()
  model.collect_params().initialize(mx.init.Xavier(), ctx=mx.gpu())

  # Loss
  l2loss = gluon.loss.L2Loss()
  optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd, 'epsilon':1e-6})

  # start the training loop
  for epoch in range(num_epochs):
    for image in dataloader:
        image = image.as_in_context(ctx)

        with mx.autograd.record():
            reconstructed = model(image)
            loss = l2loss(reconstructed, image)

        loss.backward()
        optimizer.step(batch_size)
        
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, mx.nd.mean(loss).asscalar()))

  model.save_parameters(params_file)

  return model, params_file
