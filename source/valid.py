import tensorflow as tf
import os
import random
import numpy as np
import time

from utils import write_logs, get_filepath, img_read, img_resize, save_image, img_write
from model import ms_res_net
from config import config

# Parameters
scale = config.IMG.scale

def valid_parse(valid_blur_dir, valid_sharp_dir):
	blur_string = tf.read_file(valid_blur_dir)
	sharp_string = tf.read_file(valid_sharp_dir)

	blur1, h1, w1 = tf.py_func(img_read, [blur_string], [tf.uint8, tf.int32, tf.int32])
	sharp1, _, _ = tf.py_func(img_read, [sharp_string], [tf.uint8, tf.int32, tf.int32])

	blur1 = tf.reshape(blur1, [h1, w1, 3])
	sharp1 = tf.reshape(sharp1, [h1, w1, 3])

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
	sharp2 = tf.py_func(img_resize, [sharp1, (h2,w2)], [tf.uint8])
	blur2 = tf.reshape(blur2, [h2, w2, 3])
	sharp2 = tf.reshape(sharp2, [h2, w2, 3])

	blur3 = tf.py_func(img_resize, [blur1, (h3,w3)], [tf.uint8])
	sharp3 = tf.py_func(img_resize, [sharp1, (h3,w3)], [tf.uint8])
	blur3 = tf.reshape(blur3, [h3, w3, 3])
	sharp3 = tf.reshape(sharp3, [h3, w3, 3])

	blur1 = tf.image.convert_image_dtype(blur1, tf.float32)
	sharp1 = tf.image.convert_image_dtype(sharp1, tf.float32)
	blur2 = tf.image.convert_image_dtype(blur2, tf.float32)
	sharp2 = tf.image.convert_image_dtype(sharp2, tf.float32)
	blur3 = tf.image.convert_image_dtype(blur3, tf.float32)
	sharp3 = tf.image.convert_image_dtype(sharp3, tf.float32)
	
	blur1 = tf.subtract(blur1, 0.5)
	sharp1 = tf.subtract(sharp1, 0.5)
	blur2 = tf.subtract(blur2, 0.5)
	sharp2 = tf.subtract(sharp2, 0.5)
	blur3 = tf.subtract(blur3, 0.5)
	sharp3 = tf.subtract(sharp3, 0.5)
	
	return blur1, sharp1, blur2, sharp2, blur3, sharp3

def validate(valid_dir, gen_dir, logs_dir, model_dir):
	B1 = tf.placeholder(tf.float32, [1, None, None, 3])
	S1 = tf.placeholder(tf.float32, [1, None, None, 3])
	B2 = tf.placeholder(tf.float32, [1, None, None, 3])
	S2 = tf.placeholder(tf.float32, [1, None, None, 3])
	B3 = tf.placeholder(tf.float32, [1, None, None, 3])
	S3 = tf.placeholder(tf.float32, [1, None, None, 3])

	valid_blur_path = get_filepath(valid_dir, True, '.png')
	valid_sharp_path = get_filepath(valid_dir, False, '.png')
	num_valid = len(valid_blur_path)
	valid_data = tf.data.Dataset.from_tensor_slices((valid_blur_path, valid_sharp_path))
	valid_data = valid_data.map(valid_parse, num_parallel_calls=4)
	#valid_data = valid_data.map(valid_preprocess)
	valid_data = valid_data.batch(1)
	valid_iter = valid_data.make_one_shot_iterator()
	blur1, sharp1, blur2, sharp2, blur3, sharp3 = valid_iter.get_next()

	# Prediction
	with tf.name_scope('ResNet_Deblur'):
		L1, L2, L3 = ms_res_net(B1, B2, B3, is_training=False, reuse=False)

	B1_ = tf.add(B1, 0.5)
	S1_ = tf.add(S1, 0.5)
	L1_ = tf.add(L1, 0.5)

	B1_ = tf.multiply(B1_, 255.0)
	S1_ = tf.multiply(S1_, 255.0)
	L1_ = tf.multiply(L1_, 255.0)

	B1_ = tf.clip_by_value(B1_, 0.0, 255.0)
	S1_ = tf.clip_by_value(S1_, 0.0, 255.0)
	L1_ = tf.clip_by_value(L1_, 0.0, 255.0)

	B1_ = tf.cast(B1_, tf.uint8)
	S1_ = tf.cast(S1_, tf.uint8)
	L1_ = tf.cast(L1_, tf.uint8)

	psnr = tf.image.psnr(S1_[0][:,:,:], L1_[0][:,:,:], max_val=255)
	ssim = tf.image.ssim(S1_[0][:,:,:], L1_[0][:,:,:], max_val=255)
	ms_ssim = tf.image.ssim_multiscale(S1_[0][:,:,:], L1_[0][:,:,:], max_val=255)

	# Loss function
	loss = tf.losses.mean_squared_error(S1, L1) + tf.losses.mean_squared_error(S2, L2) + tf.losses.mean_squared_error(S3, L3)

	saver = tf.train.Saver(tf.global_variables())

	with tf.Session() as sess:
		# Initialize
		sess.run(tf.global_variables_initializer())

		# Restore weights of model
		saver.restore(sess, model_dir)

		# Validation
		log = "\n========== Validation Begin ==========\n"
		write_logs(logs_dir, log, True)
		valid_start = time.time()
		avg_loss = 0
		avg_psnr = 0
		avg_ssim = 0
		avg_ms_ssim = 0
		for path in valid_blur_path:
			valid_img_start = time.time()
			bi1, si1, bi2, si2, bi3, si3 = sess.run([blur1, sharp1, blur2, sharp2, blur3, sharp3])
			feed_dict = {B1:bi1, S1:si1, B2:bi2, S2:si2, B3:bi3, S3:si3}
			_, deblur_pred, loss_val, psnr_val, ssim_val, ms_ssim_val = sess.run([L1, L1_, loss, psnr, ssim, ms_ssim], feed_dict=feed_dict)
			
			avg_loss += loss_val
			avg_psnr += psnr_val
			avg_ssim += ssim_val
			avg_ms_ssim += ms_ssim_val

			_, _, _, scene, _, img_name = path.split("\\")
			img_dir = gen_dir + "\\" + scene + "\\" + img_name
			img_write(img_dir, deblur_pred, 'PNG-FI')

			valid_img = scene + "/" + img_name
			log = "Image {}, Time {:2.5f}, Shape = {}, Loss = {:2.5f}, PSNR = {:2.5f} dB, SSIM = {:2.5f}, MS-SSIM = {:2.5f}".format(valid_img, time.time()-valid_img_start, deblur_pred.shape, loss_val, psnr_val, ssim_val, ms_ssim_val)
			write_logs(logs_dir, log, False)
		log = "\nAverage Loss = {:2.5f}".format(avg_loss/num_valid)
		write_logs(logs_dir, log, False)
		log = "Average PSNR = {:2.5f} dB".format(avg_psnr/num_valid)
		write_logs(logs_dir, log, False)
		log = "Average SSIM = {:2.5f}".format(avg_ssim/num_valid)
		write_logs(logs_dir, log, False)
		log = "Average MS-SSIM = {:2.5f}\n".format(avg_ms_ssim/num_valid)
		write_logs(logs_dir, log, False)
		log = "\nValidation Time: {:2.5f}".format(time.time()-valid_start)
		write_logs(logs_dir, log, False)
		log = "\n========== Validation End ==========\n"
		write_logs(logs_dir, log, False)
		sess.close()
    

