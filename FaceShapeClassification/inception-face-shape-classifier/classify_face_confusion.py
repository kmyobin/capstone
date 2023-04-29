#Similar to classify_face.py but generates a text file of results and a confusion matrix

import subprocess
from PIL import Image
import matplotlib.pyplot as plt
import pathlib
from datetime import datetime
import time
import tensorflow as tf, sys
import pandas as pd
import os
import numpy as np

def plot_images(image, Caption1):

	#plt.close()
	
	plt.rcParams['text.usetex'] = False
	plt.rcParams['font.size'] = 10
	plt.rcParams['font.family'] = 'Arial'
	
	fig, ax = plt.subplots(1, 1)
	ax.imshow(image)
	xlabel = Caption1
	ax.set_xlabel(xlabel)
	ax.set_xticks([])
	ax.set_yticks([])
	plt.show()
	
def classify_image(image_path, model_path, labels_path):
	# Read in the image_data
	time_start = time.monotonic()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)

# FastGFIle->GFile로 수정
	image_data = tf.io.gfile.GFile(image_path, 'rb').read()

	# Loads label file, strips off carriage return
	label_lines = [line.rstrip() for line 
					   in tf.gfile.GFile(labels_path)]

	# Unpersists graph from file
	# FastGFIle->GFile로 수정
	with tf.io.gfile.GFile(model_path, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

	with tf.Session() as sess:
		# Feed the image_data as input to the graph and get first prediction
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
		
		predictions = sess.run(softmax_tensor, \
				 {'DecodeJpeg/contents:0': image_data})
	
		# Sort to show labels of first prediction in order of confidence
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
	
		print(top_k, label_lines)
		output_label = ""
		
		for node_id in top_k:
			human_string = label_lines[node_id]
			score = predictions[0][node_id]
			output_label = output_label + human_string + "({0:.4f})".format(score) + " "
		output_label = output_label + " Runtime: " + "{0:.2f}".format(time.monotonic()-time_start) + "s"
	
	image = Image.open(image_path)
	plot_images(image,output_label)
	sess.close()
	
# change this as you see fit; model_dir is the folder containing the retrained_graph.pb and retrained_labels.txt files generated by retrain.py; imagedir contains subfolders that contain the images to be assessed

model_dir = "C:/seeun/Git/capstone/FaceShapeClassification/inception-face-shape-classifier/face_shape_celebs3_aug_500"
test_name = "celebs_extra_sorted"
imagedir = "C:/seeun/Git/capstone/FaceShapeClassification/inception-face-shape-classifier/celebs3_squared/" + test_name
results = model_dir + "/results_" + test_name + ".txt"
title = model_dir + "/confusion_" + test_name + ".txt"
result_dir = model_dir + "/result_summary_" + test_name + ".txt"

model_path = model_dir + "/retrained_graph.pb"
labels_path = model_dir + "/retrained_labels.txt"

batch_run = 1

if (batch_run == 1):
	# Read in the image_data
	time_start = time.monotonic()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)

	# Loads label file, strips off carriage return
	label_lines = [line.rstrip() for line 
						in tf.io.gfile.GFile(labels_path)]

	# Unpersists graph from file
	# FastGFIle->GFile로 수정
	with tf.io.gfile.GFile(model_path, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

	with tf.Session() as sess:
	
		# Feed the image_data as input to the graph and get first prediction
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
		sub_dir = [q for q in pathlib.Path(imagedir).iterdir() if q.is_dir()]
		confusion = np.zeros((6,8))
	
		result_summary = []
		labels = ['heart', 'oblong', 'oval', 'round', 'square']
		for j in range(len(sub_dir)):
			print("j = ",j, " ", sub_dir[j])
			images_dir = [p for p in pathlib.Path(sub_dir[j]).iterdir() if p.is_file()]
			for i in range(len(images_dir)):
				image_path = str(images_dir[i])
				# FastGFIle->GFile로 수정
				image_data = tf.io.gfile.GFile(image_path, 'rb').read()
				predictions = sess.run(softmax_tensor, \
						{'DecodeJpeg/contents:0': image_data})
	
				# Sort to show labels of first prediction in order of confidence
				top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
	
				confusion[j][int(top_k[0])] = confusion[j][int(top_k[0])] + 1
				txt = str(images_dir[i]) + " " + label_lines[top_k[0]] + " " + str(top_k[0])+ " " + str(top_k[1])+ " " + str(top_k[2])+ " " + str(top_k[3])+ " " + str(top_k[4]) + "\n"
			
				with open(results,'a') as f_handle:
					f_handle.write(txt)
				output_label = ""
				
				temp = []
				temp.append(images_dir[i])
				temp.append(labels[j])
				temp.append(label_lines[top_k[0]])
				result_summary.append(temp)
				
			f_handle.close()		
			print(confusion)
		
		#Compute the per-class and overall accuracies; some values may be incorrect because only the maximum detections per class is assumed to be the correct class label (retrain.py jumbles the ordering of labels)
		
		i,j = np.shape(confusion)
		for k in range(i):
			confusion[k][5] = np.amax(confusion[k])
			confusion[k][6] = np.sum(confusion[k][0:5])
			if confusion[k][6] != 0: confusion[k][7] = confusion[k][5]*100/confusion[k][6]
		confusion[5][5] = np.sum(confusion[0:5,5])
		confusion[5][6] = np.sum(confusion[0:5,6])
		if confusion[5][6] != 0: confusion[5][7] = confusion[k][5]*100/confusion[k][6]

		print(confusion)
		np.savetxt(title, confusion)
		np.savetxt(result_dir, result_summary, fmt = '%s')
		sess.close()

if (batch_run == 0): #single run
		
	print("Processing ", image_path, "...", end=' ', sep='') 
	out_dir = out_path + "/" + os.path.split(str(image_path))[-1]
	make_img_square(image_path,out_dir)
	classify_image(image_path, model_path, labels_path)

if (batch_run == 2): #mini-batch run
	images_dir = [p for p in pathlib.Path(out_path).iterdir() if p.is_file()]
	for i in range(len(images_dir)):
		image_path = str(images_dir[i])
		print("Processing ", image_path, "...", end=' ', sep='') 
		out_dir = out_path + "/" + os.path.split(str(image_path))[-1]
		#make_img_square(image_path,out_dir)
		classify_image(image_path, model_path, labels_path)