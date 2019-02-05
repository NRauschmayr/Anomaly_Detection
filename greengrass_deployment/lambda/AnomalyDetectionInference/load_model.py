import mxnet as mx
from mxnet import gluon

class ConvolutionalAutoencoder(gluon.nn.HybridBlock):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        with self.name_scope():
            self.encoder = gluon.nn.HybridSequential(prefix="")
            with self.encoder.name_scope():
                self.encoder.add(gluon.nn.Conv2D(32, 5, padding=0, activation='relu'))
                self.encoder.add(gluon.nn.MaxPool2D(2))
                self.encoder.add(gluon.nn.Conv2D(32, 5, padding=0, activation='relu'))
                self.encoder.add(gluon.nn.MaxPool2D(2))
                self.encoder.add(gluon.nn.Dense(2000))

            self.decoder = gluon.nn.HybridSequential(prefix="")
            with self.decoder.name_scope():
                self.decoder.add(gluon.nn.Dense(32*22*22, activation='relu'))
                self.decoder.add(gluon.nn.HybridLambda(lambda F, x: F.UpSampling(x, scale=2, sample_type='nearest')))
                self.decoder.add(gluon.nn.Conv2DTranspose(32, 5, activation='relu'))
                self.decoder.add(gluon.nn.HybridLambda(lambda F, x: F.UpSampling(x, scale=2, sample_type='nearest')))
                self.decoder.add(gluon.nn.Conv2DTranspose(1, kernel_size=5, activation='sigmoid'))


    def hybrid_forward(self, F, x):
        x = self.encoder(x)
        x = self.decoder[0](x)
        x = x.reshape((-1,32,22,22))
        x = self.decoder[1:](x)

        return x

def load_model():
  model = ConvolutionalAutoencoder()
  model.load_parameters("/home/nvidia/Projects/rauscn/model.params", ctx=mx.gpu())
  return model


