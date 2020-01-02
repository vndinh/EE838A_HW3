import tensorflow as tf
from config import config

scale = int(1.0/config.IMG.scale)

def res_net(x, is_training, reuse, name):
	x = tf.layers.conv2d(x, 64, 5, 1, 'same', activation=tf.nn.relu, name='%s/k5n64s1'%name, reuse=reuse)

	for i in range(9):
		xx = tf.layers.conv2d(x, 64, 3, 1, 'same', activation=tf.nn.relu, name='%s/res_blk_%s/k3n64s1_c1'%(name,i), reuse=reuse)
		xx = tf.layers.conv2d(xx, 64, 3, 1, 'same', name='%s/res_blk_%s/k3n64s1_c2'%(name,i), reuse=reuse)
		xx = tf.add(x, xx)
		x = xx

	x = tf.layers.conv2d(x, 3, 5, 1, 'same', name='%s/k5n3s1'%name, reuse=reuse)
	return x

def upsampling(x, scale, reuse, name):
	#x = tf.layers.conv2d_transpose(x, 3, 3, 2, 'same', name=name, reuse=reuse)
	x = tf.layers.conv2d(x, 12, 5, 1, 'same', name='%s/conv'%name, reuse=reuse)
	x = tf.depth_to_space(x, scale)
	return x

def ms_res_net(B1, B2, B3, is_training, reuse):
	L3 = res_net(B3, is_training, reuse, 'res_net_3')
	L3_ = upsampling(L3, scale, reuse, 'upsampling1')
	B2_ = tf.concat([B2, L3_], axis=3, name='concat1')

	L2 = res_net(B2_, is_training, reuse, 'res_net_2')
	L2_ = upsampling(L2, scale, reuse, 'upsampling2')
	B1_ = tf.concat([B1, L2_], axis=3, name='concat2')

	L1 = res_net(B1_, is_training, reuse, 'res_net_1')
	
	return L1, L2, L3

