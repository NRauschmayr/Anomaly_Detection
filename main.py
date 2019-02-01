import convAE
import convSTAE
import convLSTMAE
import utils
import mxnet as mx

train_directory = "UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/*/*"
test_directory  = "UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/*/*"
ctx = mx.gpu()
batch_size = 8
num_epochs = 1

#convolutional autoencoder (batch_size, 1, 100, 100)
model, params_file = convAE.train(batch_size, ctx, num_epochs, path=train_directory)
utils.plot_images( test_directory, model, params_file, ctx, output_path="convAE")

#convolutional autoencoder with stacked frames (batch_size, 10, 227, 227)
model, params_file = convSTAE.train(batch_size, ctx, num_epochs, path=train_directory)
utils.plot_images( test_directory, model, params_file, ctx, output_path="convSTAE", stacked=True)

#convolutional autoencoder with stacked frames and convolutional LSTMs (batch_size, 10, 227, 227)
model, params_file = convLSTMAE.train(batch_size, ctx, num_epochs, path=train_directory)
utils.plot_images( test_directory, model, params_file, ctx, output_path="convLSTMAE", stacked=True, lstm=True)
