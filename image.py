import socket
import time
import csv
import os
import tensorflow as tf

print(" ")
print("=========================")
print(" LOADING INCEPTION MODEL ")
print("=========================")
print(" ")
print("---------------------")
print(" TENSORFLOW WARNINGS ")
print("---------------------")
label_lines = [line.rstrip() for line in tf.gfile.GFile("models/inception_labels.txt")]
with tf.gfile.FastGFile("models/inception_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
sess = tf.Session()

# Setup client socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 8888))

s.send("image".encode())
parse = s.recv(1024).decode()
if parse == "yes" or "no":
    print("---------------------")
    print(" ")
    print("Authorisation Success.")
    print("Image Classifier Ready.")
    start = time.time()

    while True:
        data = s.recv(1024).decode()
        start = time.time()
        print(" ")
        print("==================")
        print(" REQUEST RECEIVED ")
        print("==================")
        print(" ")
        print("Request:", data)
        if parse == 'yes':
            [label, _] = data.split("-")
            path = "dataset/jpg/" + label + "/" + data + ".jpg"
        else:
            path = "predict/image.jpg"
        print("Path:", path)
        print(" ")
        print("Running Inception Model...")
        jpg_data = tf.gfile.FastGFile(path, 'rb').read()
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': jpg_data})
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        probability = {}
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            probability[human_string] = score   
        print(" ")
        print(probability)
        end = time.time()
        process_time = end-start
        print("Time Taken:", process_time)
        print(" ")
        reply = str(probability['plastic']) + " " + str(probability['metal']) + " " + str(probability['glass']) + " - " + str(process_time)
        print("Reply:", reply)
        s.send(reply.encode())
        print(" ")
        print("==================")
        print(" REQUEST COMPLETE ")
        print("==================")
        print(" ")
        print("Ready for another request.")
       
else:
    print("Authorisation Failure.")
    sys.exit("fail")