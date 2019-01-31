import convAE
import utils
import mxnet as mx

ae, params_file = convAE.train(32, mx.gpu(), 15, path='../UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/*/*')
utils.plot_images("../UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test02?/*", "convAE", ae, params_file, mx.gpu())
