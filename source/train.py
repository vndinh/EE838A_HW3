import tensorflow as tf
import os
import random
import time
import imageio

from config import config
from utils import write_logs, get_filepath, img_resize, img_read
from model import ms_res_net
from valid import valid_parse

# Directories
model_dir = config.TRAIN.model_dir
logs_dir = config.TRAIN.logs_dir
logs_train = config.TRAIN.logs_train
train_path = config.TRAIN.datapath

# Parameters
num_epoches = config.TRAIN.num_epoches
patch_size = config.TRAIN.patch_size
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.learning_rate_init
lr_decay = config.TRAIN.learning_rate_decay
lr_decay_period = config.TRAIN.decay_period

scale = config.IMG.scale

ps1 = patch_size
ps2 = int(ps1*scale)
ps3 = int(ps2*scale)

def train_parse(train_blur_dir, train_sharp_dir):
	blur_string = tf.read_file(train_blur_dir)
	sharp_string = tf.read_file(train_sharp_dir)
	
	blur_decoded = tf.image.decode_png(blur_string, channels=3)
	sharp_decoded = tf.image.decode_png(sharp_string, channels=3)
	img = tf.concat([blur_decoded, sharp_decoded], axis=2)
	
	return img

def train_preprocess(img):
	patches = tf.random_crop(img, [patch_size, patch_size, 6])
	patches = tf.image.random_flip_left_right(patches)
	patches = tf.image.random_flip_up_down(patches)
	
	blur1, sharp1 = tf.split(patches, 2, axis=2)
	
	blur2 = tf.py_func(img_resize, [blur1, (ps2,ps2)], [tf.uint8])
	sharp2 = tf.py_func(img_resize, [sharp1, (ps2,ps2)], [tf.uint8])

	blur2 = tf.reshape(blur2, [ps2, ps2, 3])
	sharp2 = tf.reshape(sharp2, [ps2, ps2, 3])

	blur3 = tf.py_func(img_resize, [blur1, (ps3,ps3)], [tf.uint8])
	sharp3 = tf.py_func(img_resize, [sharp1, (ps3,ps3)], [tf.uint8])
	blur3 = tf.reshape(blur3, [ps3, ps3, 3])
	sharp3 = tf.reshape(sharp3, [ps3, ps3, 3])
	
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

def training():
	B1 = tf.placeholder(tf.float32, [None, None, None, 3])
	S1 = tf.placeholder(tf.float32, [None, None, None, 3])
	B2 = tf.placeholder(tf.float32, [None, None, None, 3])
	S2 = tf.placeholder(tf.float32, [None, None, None, 3])
	B3 = tf.placeholder(tf.float32, [None, None, None, 3])
	S3 = tf.placeholder(tf.float32, [None, None, None, 3])

	train_blur_path = get_filepath(train_path, True, '.png')
	train_sharp_path = get_filepath(train_path, False, '.png')
	num_train = len(train_sharp_path)
	train_data = tf.data.Dataset.from_tensor_slices((train_blur_path, train_sharp_path))
	train_data = train_data.shuffle(num_train)
	train_data = train_data.map(train_parse, num_parallel_calls=4)
	train_data = train_data.map(train_preprocess, num_parallel_calls=4)
	train_data = train_data.batch(batch_size)
	train_iter = train_data.make_initializable_iterator()
	blur1, sharp1, blur2, sharp2, blur3, sharp3 = train_iter.get_next()

	valid_blur_path = ["..\\data\\valid\\GOPR0384_11_00\\blur_gamma\\000001.png",
										"..\\data\\valid\\GOPR0384_11_05\\blur_gamma\\004001.png",
										"..\\data\\valid\\GOPR0385_11_01\\blur_gamma\\003011.png"]
	valid_sharp_path = ["..\\data\\valid\\GOPR0384_11_00\\sharp\\000001.png",
											"..\\data\\valid\\GOPR0384_11_05\\sharp\\004001.png",
											"..\\data\\valid\\GOPR0385_11_01\\sharp\\003011.png"]
	valid_data = tf.data.Dataset.from_tensor_slices((valid_blur_path, valid_sharp_path))
	valid_data = valid_data.map(valid_parse, num_parallel_calls=4)
	valid_data = valid_data.batch(3)
	valid_iter = valid_data.make_initializable_iterator()
	vb1, vs1, vb2, vs2, vb3, vs3 = valid_iter.get_next()

	with tf.name_scope('ResNet_Deblur'):
		L1, L2, L3 = ms_res_net(B1, B2, B3, is_training=True, reuse=False)

	with tf.name_scope('MS_Loss'):
		loss = tf.losses.mean_squared_error(S1, L1) + tf.losses.mean_squared_error(S2, L2) + tf.losses.mean_squared_error(S3, L3)
	sum_loss_op = tf.summary.scalar("Loss", loss)

	with tf.variable_scope('learning_rate'):
		lr_v = tf.Variable(lr_init, trainable=False)

	optimizer = tf.train.AdamOptimizer(lr_v)
	gvs = optimizer.compute_gradients(loss)
	capped_gvs = [(tf.clip_by_value(grad,-0.5, 0.5), var) for grad, var in gvs]
	train_op = optimizer.apply_gradients(capped_gvs)

	saver = tf.train.Saver()
	
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

	sum_blur_op = tf.summary.image("Blur", B1)
	sum_sharp_op = tf.summary.image("Sharp", S1)
	sum_pred_op = tf.summary.image("Prediction", L1_)
	sum_im_op = tf.summary.merge([sum_blur_op, sum_sharp_op, sum_pred_op])

	psnr0 = tf.image.psnr(S1_[0][:,:,:], L1_[0][:,:,:], max_val=255)
	psnr1 = tf.image.psnr(S1_[1][:,:,:], L1_[1][:,:,:], max_val=255)
	psnr2 = tf.image.psnr(S1_[2][:,:,:], L1_[2][:,:,:], max_val=255)
	sum_psnr_op0 = tf.summary.scalar("PSNR0", psnr0)
	sum_psnr_op1 = tf.summary.scalar("PSNR1", psnr1)
	sum_psnr_op2 = tf.summary.scalar("PSNR2", psnr2)
	sum_psnr_op = tf.summary.merge([sum_psnr_op0, sum_psnr_op1, sum_psnr_op2])

	ssim0 = tf.image.ssim(S1_[0][:,:,:], L1_[0][:,:,:], max_val=255)
	ssim1 = tf.image.ssim(S1_[1][:,:,:], L1_[1][:,:,:], max_val=255)
	ssim2 = tf.image.ssim(S1_[2][:,:,:], L1_[2][:,:,:], max_val=255)
	sum_ssim_op0 = tf.summary.scalar("SSIM0", ssim0)
	sum_ssim_op1 = tf.summary.scalar("SSIM1", ssim1)
	sum_ssim_op2 = tf.summary.scalar("SSIM2", ssim2)
	sum_ssim_op = tf.summary.merge([sum_ssim_op0, sum_ssim_op1, sum_ssim_op2])

	ms_ssim0 = tf.image.ssim_multiscale(S1_[0][:,:,:], L1_[0][:,:,:], max_val=255)
	ms_ssim1 = tf.image.ssim_multiscale(S1_[1][:,:,:], L1_[1][:,:,:], max_val=255)
	ms_ssim2 = tf.image.ssim_multiscale(S1_[2][:,:,:], L1_[2][:,:,:], max_val=255)
	sum_ms_ssim_op0 = tf.summary.scalar("MS-SSIM0", ms_ssim0)
	sum_ms_ssim_op1 = tf.summary.scalar("MS-SSIM1", ms_ssim1)
	sum_ms_ssim_op2 = tf.summary.scalar("MS-SSIM2", ms_ssim2)
	sum_ms_ssim_op = tf.summary.merge([sum_ms_ssim_op0, sum_ms_ssim_op1, sum_ms_ssim_op2])

	if num_train % batch_size != 0:
		num_batches = int(num_train/batch_size) + 1
	else:
		num_batches = int(num_train/batch_size)

	with tf.Session() as sess:
		# Initialize variables
		sess.run(tf.global_variables_initializer())

		# Op to write logs to Tensorboard
		train_sum_writer = tf.summary.FileWriter(logs_dir, tf.get_default_graph())

		# Training process
		log = "\n========== Training Begin ==========\n"
		write_logs(logs_train, log, True)
		train_start = time.time()
		for epoch in range(num_epoches):
			epoch_start = time.time()
			
			if (epoch > 399) and (epoch % lr_decay_period == 0):
				new_lr = lr_v * lr_decay
				sess.run(tf.assign(lr_v, new_lr))
				log = "** New learning rate: %1.9f **\n" % (lr_v.eval())
				write_logs(logs_train, log, False)
			elif epoch == 0:
				sess.run(tf.assign(lr_v, lr_init))
				log = "** Initial learning rate: %1.9f **\n" % (lr_init)
				write_logs(logs_train, log, False)
			
			avg_loss = 0
			sess.run(train_iter.initializer)
			for batch in range(num_batches):
				batch_start = time.time()
				bp1, sp1, bp2, sp2, bp3, sp3 = sess.run([blur1, sharp1, blur2, sharp2, blur3, sharp3])
				train_dict = {B1:bp1, S1:sp1, B2:bp2, S2:sp2, B3:bp3, S3:sp3}
				_, loss_val, sum_loss = sess.run([train_op, loss, sum_loss_op], feed_dict=train_dict)
				avg_loss += loss_val
				train_sum_writer.add_summary(sum_loss, epoch*num_batches+batch)
				log = "Epoch {}, Time {:2.5f}, Batch {}, Batch Loss = {:2.5f}".format(epoch, time.time()-batch_start, batch, loss_val)
				write_logs(logs_train, log, False)
			log = "\nEpoch {}, Time {:2.5f}, Average Loss = {:2.5f}\n".format(epoch, time.time()-epoch_start, avg_loss/num_batches)
			write_logs(logs_train, log, False)

			sess.run(valid_iter.initializer)
			bi1, si1, bi2, si2, bi3, si3 = sess.run([vb1, vs1, vb2, vs2, vb3, vs3])
			valid_dict = {B1:bi1, S1:si1, B2:bi2, S2:si2, B3:bi3, S3:si3}
			_, sum_im, sum_psnr, sum_ssim, sum_ms_ssim = sess.run([L1, sum_im_op, sum_psnr_op, sum_ssim_op, sum_ms_ssim_op], feed_dict=valid_dict)
			train_sum_writer.add_summary(sum_im, epoch)
			train_sum_writer.add_summary(sum_psnr, epoch)
			train_sum_writer.add_summary(sum_ssim, epoch)
			train_sum_writer.add_summary(sum_ms_ssim, epoch)

		log = "\nTraining Time: {}".format(time.time()-train_start)
		write_logs(logs_train, log, False)
		log = "\n========== Training End ==========\n"
		write_logs(logs_train, log, False)

		# Save model
		save_path = saver.save(sess, model_dir)
		log = "Model is saved in file: %s" % save_path
		write_logs(logs_train, log, False)
		log = "Run the command line:\n" \
					"--> tensorboard --logdir=../logs"
		write_logs(logs_train, log, False)
		sess.close()

