import numpy as np
import glob
from PIL import Image

MY_S3_BUCKET = ""

files = sorted(glob.glob('UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/*/*'))

a = np.zeros( (int(len(files)/10), 10, 227, 227))

i,idx = 0,0
for filename in range(0, len(files)):
    im = Image.open(files[filename])
    im = im.resize((n,n))
    a[idx,i,:,:] = np.array(im, dtype=np.float32)/255.0
    i = i + 1
    if i >= time:
      idx = idx + 1
      i = 0

np.save("input_data", a)
inputs = sagemaker_session.upload_data(path='input_data.npy', bucket=MY_S3_BUCKET, key_prefix='data')

num_splits = 4
split_index = 25

for i in range(0,num_splits):
    upload = a[i*split_index:(i+1)*split_index]
    local_path = "input_data" + str(i) +".npy"
    np.save(local_path, upload)
    inputs = sagemaker_session.upload_data(path=local_path, bucket=MY_S3_BUCKET, key_prefix='data-split')
