# EE838A_HW3
Homework 3, Advanced Image Restoration and Quality Enhancement, EE, KAIST, Fall 2018

1. Explaination
	- The '../data' folder contains the training and validation dataset
	- All informations during training, validation, testing are recorded in '../logs'
	- The '../model' includes the weights of the network after finishing training
	- The outputs of validation is saved in '../report/valid_deblur_gen'
	- The outputs of testing is saved in '../report/test_deblur_gen'
	- The model using pixel shuffle for up-convolution is saved in '../report/model_pixel_shuffle'
	- The model using transposed convolution is saved in '../report/model_trans_conv'
	- All coding files are written in '../source'
	- For more detail explaination, read '../report/HW3_20184187_DinhVu_Report.pdf'

2. Choose the model
	- If you choose the network using pixel shuffle for up-convolution:
		+ Copy all files in '../report/model_pixel_shuffle' to '../model'
		+ In the file '../source/model.py', uncomment line 20-21 and comment line 19
		+ For easier understand,see Figure 2.2 in '../report/HW3_20184187_DinhVu_Report.pdf'

	- If you choose the network using transposed convolution for up-convolution:
		+ Copy all files in '../report/model_trans_conv' to '../model'
		+ In the file '../source/model.py', comment line 20-21 and uncomment line 19
		+ For easier understand, see Figure 2.3 in '../report/HW3_20184187_DinhVu_Report.pdf'

3. Training
	- Copy the given training dataset to '../data/train'
	- Open a Command Promt (Windows) or Terminal (Linux) in the folder '../source'
	- Type: python main.py --mode=train
	- Wait the training process finish
	- If you want using Tensorboard to see what happend during training:
		+ Open another Command Promt (Windows) or Terminal (Linux) in the folder '../source'
		+ Type: tensorboard --logdir=../logs
		+ Follow the instruction appeared after that

4. Validation
	- Copy the given validation dataset to '../data/valid'
	- Open a Command Promt (Windows) or Terminal (Linux) in the folder '../source'
	- Type: python main.py --mode=valid
	- Wait the validation process finish
	- See the prediction images in '../report/valid_deblur_gen' folfer
	- Read the arguments of each validation image in '../logs/logs_valid.txt'

5. Testing
	- Copy the testing images to '../data/test'
	- Open a Command Promt (Windows) or Terminal (Linux) in the folder '../source'
	- Type: python main.py --mode=test
	- Wait the testing process finish
	- See the prediction images in '../report/test_deblur_gen' folder
	- Read the arguments of each test image in '../logs/logs_test.txt'

