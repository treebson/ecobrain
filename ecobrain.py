import socket
import base64
import os
import select
import glob
import numpy as np
import tensorflow as tf
import sys
import time
import csv
from sklearn.model_selection import train_test_split

threshold = 0.99

host = ""
port = 8888
path = "C:/Users/tyson/Desktop/EcoRobotics/Product/cloud/"

conn_list = []
conn_model = {}
model_conn = {}

print(" ")
parse = input("Parse (yes/no): ")

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('localhost', 8888))
s.listen(3)
connected_models = 0
print(" ")
print("Waiting for all classifiers to connect.")
while connected_models < 3:
	conn, _ = s.accept()
	conn_list.append(conn)
	data = conn.recv(1024).decode()
	if data == "image" or data == "sound" or data == "series":
		connected_models += 1
		reply = parse
		conn_model[conn] = data
		model_conn[data] = conn
		if data == "image":
			print("Image Classifier has joined.")
		elif data == "sound":
			print("Sound Classifier has joined.")
		else:
			print("Series Classifier has joined.")
	else:
		reply = "declined"
	conn.send(reply.encode())
print(" ")
print("All classifiers connected!")	

def setupServer():
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	try:
		s.bind((host, port))
		IP = [i for i in socket.gethostbyname_ex(socket.gethostname())[2] if not i.startswith("127.")][:1]
		IP = ''.join(IP)
		IP = str(IP)
		print(" ")
		print("==================")
		print(" ECOSERVER ONLINE ")
		print("==================")
		print(" ")
		print("Server IP: " + IP)
		print(" ")
	except socket.error as msg:
		print(msg)
	return s

def setupConnection():
	print("Awaiting client to connect.")
	s.listen(1)	# Allows one connection at a time.
	conn, address =  s.accept()
	print("Connected to client.")
	print("Client IP: " + address[0])
	print(" ")
	return conn

def parse_dataset():
	print(" ")
	print("====================")
	print(" COLLECTING DATASET ")
	print("====================")
	print(" ")
	features = []
	labels = []
	label_encoding = {"plastic": 0, "metal": 1, "glass": 2, "paper": 3}
	for file_path in glob.glob("dataset/jpg/*/*.jpg"):
		jpg_file = file_path
		wav_file = jpg_file.replace('jpg', 'wav')
		csv_file = jpg_file.replace('jpg', 'csv')
		[_, _, file_name] = jpg_file.split('\\') 
		file_name, _ = file_name.split('.')
		label, _ = file_name.split('-')
		model_conn["image"].send(file_name.encode())
		model_conn["sound"].send(file_name.encode())
		model_conn["series"].send(file_name.encode())
		model_reply = {}
		completed_models = 0
		while completed_models < 3:
			completed, _, _ = select.select(conn_list, [], [])
			for c in completed:
				completed_models += 1
				data = c.recv(1024).decode()
				model_reply[conn_model[c]] = data
		image_features, _ = model_reply['image'].split(" - ")
		sound_features, _ = model_reply['sound'].split(" - ")
		series_features, _ = model_reply['series'].split(" - ")
		image_features = image_features.split(" ")
		sound_features = sound_features.split(" ")
		series_features = series_features.split(" ")
		feature_vector = []
		feature_vector.extend(image_features)
		feature_vector.extend(sound_features)
		feature_vector.extend(series_features)
		feature_vector = list(map(float,feature_vector))
		features.append(feature_vector)
		labels.append(label_encoding[label])
		test_features = np.array(features)
		test_labels = np.array(labels)
		print("Features:", test_features.shape)
		print("Labels:", test_labels.shape)

	features = np.array(features)
	labels = np.array(labels)
	# SAVE
	np.save('processed/combined_features.npy', features)
	np.save('processed/combined_labels.npy', labels)	

if parse == "yes":
	parse_dataset()

n_nodes = 50
#n_nodes = 20
n_epochs = 10
learning_rate = 0.01
n_classes = 3
batch_size = 10

print(" ")
print("==============================")
print(" TRAINING COMBINED CLASSIFIER ")
print("==============================")
print(" ")
print("Load Dataset...")
print(" ")
features = np.load('processed/combined_features.npy')
labels = np.load('processed/combined_labels.npy')
labels = np.eye(n_classes)[labels]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0, random_state=42)
print("Training Set:")
print("X_train", X_train.shape)
print("Y_train", y_train.shape)
print(" ")
print("Testing Set:")
print("X_train", X_test.shape)
print("Y_train", y_test.shape)
print(" ")
print("---------------------")
print(" TENSORFLOW WARNINGS ")
print("---------------------")

n_dim = X_train.shape[1]
sd = 1 / np.sqrt(n_dim)
x = tf.placeholder(tf.float32, [None, n_dim], name="x")
y = tf.placeholder(tf.float32, [None, n_classes], name="y")

def combined_neural_network_model(data):
	keep_prob = tf.placeholder(tf.float32)
	hidden_layer = {'W': tf.Variable(tf.random_normal([n_dim, n_nodes], mean=0, stddev=sd)),
					'b': tf.Variable(tf.random_normal([n_nodes], mean=0, stddev=sd))}
	output_layer = {'W': tf.Variable(tf.random_normal([n_nodes, n_classes], mean=0, stddev=sd)),
					'b': tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))}
	hidden = tf.matmul(data, hidden_layer['W']) + hidden_layer['b']
	hidden = tf.nn.relu(hidden)
	dropout = tf.nn.dropout(hidden, keep_prob)
	output = tf.matmul(hidden, output_layer['W']) + output_layer['b']
	return output

def train_combined_neural_network(x):
	epoch_list = []
	loss_list = []
	logits = combined_neural_network_model(x)
	combined_classify = tf.nn.softmax(logits=logits, name="combined_classify")
	tf.add_to_collection("combined_classify", combined_classify)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
	optimizer = tf.train.AdamOptimizer()
	train = optimizer.minimize(cost)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("---------------------")
		print(" ")
		print("Training...")
		print(" ")
		start = time.time()
		for epoch in range(n_epochs):
			epoch_loss = 0
			for i in range(int(X_train.shape[0]/batch_size)):
				epoch_x = X_train[i*batch_size:(i+1)*batch_size,:]
				epoch_y = y_train[i*batch_size:(i+1)*batch_size,:]
				_, c = sess.run([train, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
			currtime = time.strftime("[%H:%M:%S]", time.gmtime())
			print(currtime, '\tEpoch #:', epoch+1, 'of', n_epochs, '\tEpoch Loss:', epoch_loss)
			epoch_list.append(epoch+1)
			loss_list.append(epoch_loss)
		end = time.time()
		train_time = end-start
		with open("performance/combined_loss.csv", 'w', newline='') as csvfile:
			fieldnames = ['Epoch', 'Loss']
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writeheader()
			for i in range(len(loss_list)):
				writer.writerow({'Epoch': epoch_list[i], 'Loss': loss_list[i]})
		correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
		train_accuracy = accuracy.eval({x: X_test, y: y_test})*100
		print(" ")
		print('Final test accuracy = %.1f%% (N=%d)' % (train_accuracy, len(y_test)))
		print(" ")
		print('Total training time = ', train_time)
		print(" ")
		with open("performance/combined_train.csv", 'a', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow([train_accuracy, train_time])
		# Save Model for later use
		saver = tf.train.Saver()
		saver.save(sess, 'models/combined_classifier')
		saver.export_meta_graph('models/combined_classifier.meta')
		print('Model Saved: models/combined_classifier')

train_combined_neural_network(x)

sess = tf.Session()
new_saver = tf.train.import_meta_graph('models/combined_classifier.meta')
new_saver.restore(sess, 'models/combined_classifier')
combined_classify = tf.get_collection('combined_classify')


def dataTransfer(conn):
	mode = receiveMessage(conn)
	if mode == "collect":
		print("=================")
		print(" COLLECTION MODE ")
		print("=================")
		label = receiveMessage(conn)
		print("Collecting " + label + ".")
		print(" ")
		while True:
			for _ in range(3):
				form = receiveMessage(conn)
				data = receiveMessage(conn)
				filepath = path + "dataset/" + form + "/" + label
				count = str(len([f for f in os.listdir(filepath)]))
				filename = filepath + "/" + label + "-" + count + "." + form
				with open(filename, "wb") as f:
					f.write(base64.b64decode(data))
				print("Received '" + filename + "'")
			print(" ")	
	if mode == "predict":
		print("=================")
		print(" PREDICTION MODE ")
		print("=================")
		print(" ")
		while True:
			print("Receiving...")
			for _ in range(3):
				form = receiveMessage(conn)
				data = receiveMessage(conn)
				if form == "jpg":
					filename = "predict/image.jpg"
				if form == "wav":
					filename = "predict/sound.wav"
				if form == "csv":
					filename = "predict/series.csv"
				with open(filename, "wb") as f:
					f.write(base64.b64decode(data))
				print("Received '" + filename + "'")
			print("Predicting...")
			model_conn["image"].send("predict/image.jpg".encode())
			model_conn["sound"].send("predict/sound.wav".encode())
			model_conn["series"].send("predict/series.csv".encode())
			model_reply = {}
			completed_models = 0
			while completed_models < 3:
				completed, _, _ = select.select(conn_list, [], [])
				for c in completed:
					completed_models += 1
					data = c.recv(1024).decode()
					model_reply[conn_model[c]] = data
			image_features, image_time = model_reply['image'].split(" - ")
			sound_features, sound_time = model_reply['sound'].split(" - ")
			series_features, series_time = model_reply['series'].split(" - ")
			image_features = image_features.split(" ")
			sound_features = sound_features.split(" ")
			series_features = series_features.split(" ")
			image_time, sound_time, series_time = float(image_time), float(sound_time), float(series_time)
			feature_vector = []
			feature_vector.extend(image_features)
			feature_vector.extend(sound_features)
			feature_vector.extend(series_features)
			feature_vector = list(map(float,feature_vector))
			probability = sess.run(combined_classify, feed_dict={x: [feature_vector]})[0][0]
			most_confident = np.argmax(probability)
			encode_label = {0: "Plastic", 1: "Metal", 2: "Glass"}
			print("Plastic:", "%.2f" % (probability[0]*100) + "%")
			print("Metal:  ", "%.2f" % (probability[1]*100) + "%")
			print("Glass:  ", "%.2f" % (probability[2]*100) + "%")
			if probability[most_confident] < threshold:
				vote = "Landfill"
				print("Confidence below 99% threshold, predicting Landfill")
			else:
				vote = encode_label[most_confident]
				print("Confidence above 99% threshold, predicting", vote)
			conn.send(vote.encode())
			print(" ")
	print("Closing connection.")
	print(" ")
	conn.close()

def receiveMessage(conn):
	data = conn.recv(8).decode()
	if not data:
		print("Error: Length not received.")
		conn.send("0".encode())
		return None
	conn.send("1".encode())
	length = int(data)
	data = ''
	#print("Receiving", length, "bytes.")
	while len(data) < length:
		packet = conn.recv(length - len(data)).decode()
		if not packet:
			print("Error: Data not received.")
			conn.send("0".encode())
			return None
		data = data + packet
	conn.send("1".encode())
	return data

#======
# MAIN
#======

s = setupServer()
while True:
	conn = setupConnection()
	dataTransfer(conn)