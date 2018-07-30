import socket
import numpy as np
import tensorflow as tf
import librosa
import time
import csv
import glob

from time import gmtime, strftime

from sklearn.model_selection import train_test_split

print(" ")
parse = input("Parse (yes/no): ")

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

def parse_dataset():
    print(" ")
    print("============================")
    print(" FEATURE EXTRACTING DATASET ")
    print("============================")
    print(" ")
    features = []
    labels = []
    # Read plastic wav data
    for wav_name in glob.glob("dataset/wav/plastic/*.wav"):
        mfccs, chroma, mel, contrast, tonnetz = extract_feature(wav_name)
        extra_features = []
        extra_features.extend(mfccs)
        extra_features.extend(chroma)
        extra_features.extend(mel)
        extra_features.extend(contrast)
        extra_features.extend(tonnetz)
        features.append(extra_features)
        labels.append(0)
        print(wav_name)
    # Read metal wav data
    for wav_name in glob.glob("dataset/wav/metal/*.wav"):
        mfccs, chroma, mel, contrast, tonnetz = extract_feature(wav_name)
        extra_features = []
        extra_features.extend(mfccs)
        extra_features.extend(chroma)
        extra_features.extend(mel)
        extra_features.extend(contrast)
        extra_features.extend(tonnetz)
        features.append(extra_features)
        labels.append(1)
        print(wav_name)
    # Read glass wav data
    for wav_name in glob.glob("dataset/wav/glass/*.wav"):
        mfccs, chroma, mel, contrast, tonnetz = extract_feature(wav_name)
        extra_features = []
        extra_features.extend(mfccs)
        extra_features.extend(chroma)
        extra_features.extend(mel)
        extra_features.extend(contrast)
        extra_features.extend(tonnetz)
        features.append(extra_features)
        labels.append(2)
        print(wav_name)
    # Read paper wav data
    '''
    for wav_name in glob.glob("dataset/wav/paper/*.wav"):
        mfccs, chroma, mel, contrast, tonnetz = extract_feature(wav_name)
        extra_features = []
        extra_features.extend(mfccs)
        extra_features.extend(chroma)
        extra_features.extend(mel)
        extra_features.extend(contrast)
        extra_features.extend(tonnetz)
        features.append(extra_features)
        labels.append(3)
        print(wav_name)
    '''
    features = np.array(features)
    labels = np.array(labels)
    # SAVE
    np.save('processed/sound_features.npy', features)
    np.save('processed/sound_labels.npy', labels)

if parse == "yes":
    parse_dataset()

# Sound Classifier Variables
n_hl1 = 300
n_hl2 = 280
#n_hl1 = 100
#n_hl2 = 50
learning_rate = 0.01
n_classes = 3
batch_size = 10
n_epochs = 10

print(" ")
print("===========================")
print(" TRAINING SOUND CLASSIFIER ")
print("===========================")
print(" ")
print("Loading Dataset...")
print(" ")
features = np.load('processed/sound_features.npy')
labels = np.load('processed/sound_labels.npy')

print("Dataset Shape:")
print("features", features.shape)
print("labels", labels.shape)
print(" ")
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0, random_state=42)
y_train = np.eye(n_classes)[y_train]
y_test = np.eye(n_classes)[y_test]
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

def neural_network_model(data):
    keep_prob = tf.placeholder(tf.float32)
    hl1 = {'W': tf.Variable(tf.random_normal([n_dim, n_hl1], mean=0, stddev=sd)),
           'b': tf.Variable(tf.random_normal([n_hl1], mean=0, stddev=sd))}
    hl2 = {'W': tf.Variable(tf.random_normal([n_hl1, n_hl2], mean=0, stddev=sd)),
           'b': tf.Variable(tf.random_normal([n_hl2], mean=0, stddev=sd))}     
    ol = {'W': tf.Variable(tf.random_normal([n_hl2, n_classes], mean=0, stddev=sd)),
          'b': tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))}
    l1 = tf.matmul(data, hl1['W']) + hl1['b']
    l1 = tf.nn.relu(l1)
    l2 = tf.matmul(l1, hl2['W']) + hl2['b']
    l2 = tf.nn.relu(l2)
    dropout = tf.nn.dropout(l2, keep_prob)
    output = tf.matmul(l2, ol['W']) + ol['b']
    return output

def train_neural_network(x):
    epoch_list = []
    loss_list = []
    # Define tensors
    logits = neural_network_model(x)
    classify = tf.nn.softmax(logits, name="classify")
    tf.add_to_collection("classify", classify)
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
            currtime = strftime("[%H:%M:%S]", gmtime())
            print(currtime, '\tEpoch #:', epoch+1, 'of', n_epochs, '\tEpoch Loss:', epoch_loss)
            epoch_list.append(epoch+1)
            loss_list.append(epoch_loss)
        end = time.time()
        train_time = end-start
        with open("performance/sound_loss.csv", 'w', newline='') as csvfile:
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
        print('Total training time =', train_time)
        print(" ")
        with open("performance/sound_train.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([train_accuracy, train_time])
        # Save model for later use
        saver = tf.train.Saver()
        saver.save(sess, 'models/sound_classifier')
        saver.export_meta_graph('models/sound_classifier.meta')
        print('Model Saved: models/sound_classifier')

train_neural_network(x)

# Setup client socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 8888))

s.send("sound".encode())
parse = s.recv(1024).decode()
if parse == "yes" or "no":
    print(" ")
    print("Authorisation Success.")
    print("Sound Classifier Ready.")
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('models/sound_classifier.meta')
    new_saver.restore(sess, 'models/sound_classifier')
    classify = tf.get_collection('classify')
    while True:
        data = s.recv(1024).decode()
        start = time.time()
        print(" ")
        print("==================")
        print(" REQUEST RECEIVED ")
        print("==================")
        print(" ")
        print("Request:", data)
        if parse == "yes":
            [label, _] = data.split("-")
            path = "dataset/wav/" + label + "/" + data + ".wav"
        else:
            path = "predict/sound.wav"
        print("Path:", path)
        print(" ")
        print("Extracting Features...")
        print(" ")
        mfccs, chroma, mel, contrast, tonnetz = extract_feature(path)
        sound_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        print("Running Multilayer Perceptron...")
        # Restore Model
        predictions = sess.run(classify, feed_dict={x: [sound_features]})[0][0]
        probability = {}
        probability['plastic'] = predictions[0]
        probability['metal'] = predictions[1]
        probability['glass'] = predictions[2]
        #probability['paper'] = predictions[3]
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
    sys.exit(0)