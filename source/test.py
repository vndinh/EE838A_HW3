import tensorflow as tf
import os
import random
import numpy as np
import time

from utils import write_logs, img_read, img_resize, save_image, img_write
from model import ms_res_net
from config import config

# Parameters
scale = config.IMG.scale

def get_test_img(path, suffix):
	img_path = []
	for f in os.listdir(path):
		if f.endswith(suffix):
			img_path.append(os.path.join(path, f))
	img_path = sorted(img_path)
	return img_path

def test_parse(test_blur_dir):
	blur_string = tf.read_file(test_blur_dir)

	blur1, h1, w1 = tf.py_func(img_read, [blur_string], [tf.uint8, tf.int32, tf.int32])

	blur1 = tf.reshape(blur1, [h1, w1, 3])

	h1 = tf.cast(h1, tf.float32)
	w1 = tf.cast(w1, tf.float32)

	h2 = scale * h1
	w2 = scale * w1
	h3 = scale * h2
	w3 = scale * w2

	h2 = tf.cast(h2, tf.int32)
	w2 = tf.cast(w2, tf.int32)
	h3 = tf.cast(h3, tf.int32)
	w3 = tf.cast(w3, tf.int32)
	
	blur2 = tf.py_func(img_resize, [blur1, (h2,w2)], [tf.uint8])
	blur2 = tf.reshape(blur2, [h2, w2, 3])

	blur3 = tf.py_func(img_resize, [blur1, (h3,w3)], [tf.uint8])
	blur3 = tf.reshape(blur3, [h3, w3, 3])

	blur1 = tf.image.convert_image_dtype(blur1, tf.float32)
	blur2 = tf.image.convert_image_dtype(blur2, tf.float32)
	blur3 = tf.image.convert_image_dtype(blur3, tf.float32)
	
	blur1 = tf.subtract(blur1, 0.5)
	blur2 = tf.subtract(blur2, 0.5)
	blur3 = tf.subtract(blur3, 0.5)

	return blur1, blur2, blur3

def testing(test_dir, gen_dir, logs_dir, model_dir):
	B1 = tf.placeholder(tf.float32, [1, None, None, 3])
	B2 = tf.placeholder(tf.float32, [1, None, None, 3])
	B3 = tf.placeholder(tf.float32, [1, None, None, 3])

	test_blur_path = get_test_img(test_dir, '.png')
	num_test = len(test_blur_path)
	test_data = tf.data.Dataset.from_tensor_slices(test_blur_path)
	test_data = test_data.map(test_parse, num_parallel_calls=4)
	test_data = test_data.batch(1)
	test_iter = test_data.make_one_shot_iterator()
	blur1, blur2, blur3 = test_iter.get_next()

	# Prediction
	with tf.name_scope('ResNet_Deblur'):
		L1, L2, L3 = ms_res_net(B1, B2, B3, is_training=False, reuse=False)

	L1_ = tf.add(L1, 0.5)
	L1_ = tf.multiply(L1_, 255.0)
	L1_ = tf.clip_by_value(L1_, 0.0, 255.0)
	L1_ = tf.cast(L1_, tf.uint8)

	saver = tf.train.Saver(tf.global_variables())

	with tf.Session() as sess:
		# Initialize
		sess.run(tf.global_variables_initializer())

		# Restore weights of model
		saver.restore(sess, model_dir)

		# Testing
		log = "\n========== Test Begin ==========\n"
		write_logs(logs_dir, log, True)
		test_start = time.time()
		for i in range(num_test):
			test_img_start = time.time()
			bi1, bi2, bi3 = sess.run([blur1, blur2, blur3])
			feed_dict = {B1:bi1, B2:bi2, B3:bi3}
			_, deblur_pred = sess.run([L1, L1_], feed_dict=feed_dict)
			img_name = "{:06d}.png".format(i)
			img_dir = gen_dir + "\\" + img_name
			img_write(img_dir, deblur_pred, 'PNG-FI')

			log = "Image {}, Time {:2.5f}, Shape = {}".format(img_name, time.time()-test_img_start, deblur_pred.shape)
			write_logs(logs_dir, log, False)
		log = "\nTest Time: {:2.5f}".format(time.time()-test_start)
		write_logs(logs_dir, log, False)
		log = "\n========== Test End ==========\n"
		write_logs(logs_dir, log, False)
		sess.close()
	

