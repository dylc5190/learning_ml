import tensorflow as tf
import numpy as np
from sklearn.datasets import load_sample_images
# Load sample images
dataset = np.array(load_sample_images().images, dtype=np.float32)
batch_size, height, width, channels = dataset.shape
height = width = 6	# reduce the size for simplicity
channels = 2
nr_filters = 2
filters = np.zeros(shape=(3, 3, channels, nr_filters), dtype=np.float32)  # I think this way: I have <nr_filters> sets of filters and each set contains <channels> filters.
filters[:, 1, 0, 0] = 1  # make channel filters different for demonstration purpose
filters[:, 1, 1, 0] = 2  # I don't know when they should be the same and when different
filters[1, :, 0, 1] = 1  # 2nd set of filters
filters[1, :, 1, 1] = 1
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filters, strides=[1,3,3,1], padding="SAME")	# produce an 2x2 feature map (6/3 = 2)
with tf.Session() as sess:
  output = sess.run(convolution, feed_dict={X: dataset[0:1,0:6,0:6,0:2]})	# 0:1 to specify batch size 1. only use 2 channels of 6x6 data.
print('I cannot figure this out as the layers of feature maps')
print(output)
print('I think this format looks better')
print(output[:,:,:,0])
print(output[:,:,:,1])


# now calculate output by myself
input_channel_1= dataset[0,0:6,0:6,0]
input_channel_2= dataset[0,0:6,0:6,1]
for index_filter_set in range(2):
 print("feature map {}".format(index_filter_set))
 for i in range(0,6,3):
  for j in range(0,6,3):
    sum_ch1 = np.sum(np.multiply(input_channel_1[i:i+3,j:j+3],filters[:, :, 0, index_filter_set])) # 1st of filter set 1: for channel 1
    sum_ch2 = np.sum(np.multiply(input_channel_2[i:i+3,j:j+3],filters[:, :, 1, index_filter_set])) # 2nd of filter set 1: for channel 2
    sum_layer = sum_ch1 + sum_ch2
    print("{} ".format(sum_layer),end='')
  print()
 print(output[:,:,:,index_filter_set])
 print()
