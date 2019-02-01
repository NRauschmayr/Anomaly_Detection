import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
import utils

# convolutional autoencoder with convolutional LSTMs
class convLSTMAE(gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
        super(convLSTMAE, self).__init__(**kwargs)
        with self.name_scope():

          self.encoder = gluon.nn.HybridSequential()
          self.encoder.add(gluon.nn.Conv2D(128, kernel_size=11, strides=4, activation='relu'))
          self.encoder.add(gluon.nn.Conv2D(64, kernel_size=5, strides=2, activation='relu'))

          self.temporal_encoder = gluon.rnn.HybridSequentialRNNCell()
          self.temporal_encoder.add(gluon.contrib.rnn.Conv2DLSTMCell((64,26,26), 64, 3, 3, i2h_pad=1))
          self.temporal_encoder.add(gluon.contrib.rnn.Conv2DLSTMCell((64,26,26), 32, 3, 3, i2h_pad=1))
          self.temporal_encoder.add(gluon.contrib.rnn.Conv2DLSTMCell((32,26,26), 64, 3, 3, i2h_pad=1))

          self.decoder =  gluon.nn.HybridSequential()
          self.decoder.add(gluon.nn.Conv2DTranspose(channels=128, kernel_size=5, strides=2, activation='relu'))
          self.decoder.add(gluon.nn.Conv2DTranspose(channels=10, kernel_size=11, strides=4, activation='sigmoid'))

    def hybrid_forward(self, F, x, states=None, **kwargs):
        x = self.encoder(x)
        x, states = self.temporal_encoder(x, states)
        x = self.decoder(x)

        return x, states

# Train the autoencoder
def train(batch_size, ctx, num_epochs, path, lr=1e-4, wd=1e-5, params_file="autoencoder_ucsd_convLSTMAE.params"):

  # Dataloader for training dataset
  dataloader = utils.create_dataset_stacked_images(path, batch_size, shuffle=True, augment=True)

  # Get model
  model = convLSTMAE()
  model.hybridize()
  model.collect_params().initialize(mx.init.Xavier(), ctx=mx.gpu())
  states = model.temporal_encoder.begin_state(batch_size=batch_size, ctx=ctx)

  # Loss
  l2loss = gluon.loss.L2Loss()
  optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd, 'epsilon':1e-6})

  # start the training loop
  for epoch in range(num_epochs):
    for image in dataloader:

        image  = image.as_in_context(ctx)
        states = model.temporal_encoder.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=ctx)
        
        with mx.autograd.record():
            reconstructed, states = model(image, states)
            loss = l2loss(reconstructed, image)

        loss.backward()
        optimizer.step(batch_size)
        
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, mx.nd.mean(loss).asscalar()))

  #save parameters
  model.save_parameters(params_file)

  return model, params_file

