import socket
import glob
import csv
import numpy as np
import time
import csv
from knndtw import KnnDtw

freqs = []
masses = []
labels = []
# Read plastic csv data
for csv_name in glob.glob("dataset/csv/plastic/*.csv"):
    with open(csv_name, "r") as f:
        reader = csv.DictReader(f)
        freq_list = []
        mass_list = []
        for row in reader:
            freq = float(row['Freq'])
            freq_list.append(freq)
            mass = float(row['Mass'])
            mass_list.append(mass)
        freqs.append(freq_list)
        masses.append(mass_list)
        labels.append(0)
# Read metal csv data
for csv_name in glob.glob("dataset/csv/metal/*.csv"):
    with open(csv_name, "r") as f:
        reader = csv.DictReader(f)
        freq_list = []
        mass_list = []
        for row in reader:
            freq = float(row['Freq'])
            freq_list.append(freq)
            mass = float(row['Mass'])
            mass_list.append(mass)
        freqs.append(freq_list)
        masses.append(mass_list)
        labels.append(1)
# Read glass csv data
for csv_name in glob.glob("dataset/csv/glass/*.csv"):
    with open(csv_name, "r") as f:
        reader = csv.DictReader(f)
        freq_list = []
        mass_list = []
        for row in reader:
            freq = float(row['Freq'])
            freq_list.append(freq)
            mass = float(row['Mass'])
            mass_list.append(mass)
        freqs.append(freq_list)
        masses.append(mass_list)
        labels.append(2)
# Read paper csv data
'''
for csv_name in glob.glob("dataset/csv/paper/*.csv"):
    with open(csv_name, "r") as f:
        reader = csv.DictReader(f)
        freq_list = []
        mass_list = []
        for row in reader:
            freq = float(row['Freq'])
            freq_list.append(freq)
            mass = float(row['Mass'])
            mass_list.append(mass)
        freqs.append(freq_list)
        masses.append(mass_list)
        labels.append(3)
'''
freqs = np.array(freqs)
masses = np.array(masses)
labels = np.array(labels)

print(" ")
print("=============================")
print(" INSTANTIATING KNN-DTW MODEL ")
print("=============================")
print(" ")
print("Loading List of Neighbours...")
print("freqs:", freqs.shape)
print("masses:", masses.shape)
print("labels:", labels.shape)
print(" ")
freq_model = KnnDtw(n_neighbors=10, max_warping_window=10)
freq_model.fit(freqs, labels)
mass_model = KnnDtw(n_neighbors=10, max_warping_window=10)
mass_model.fit(masses, labels)

# Setup client socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 8888))

s.send("series".encode())
parse = s.recv(1024).decode()
if parse == "yes" or "no":
	print("Authorisation Success.")
	print("Series Classifier Ready.")
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
			path = "dataset/csv/" + label + "/" + data + ".csv"
		else:
			path = "predict/series.csv"
		print("Path:", path)
		print(" ")
		print("Searching for KNN-DTW...")
		print(" ")
		with open(path, "r") as f:
			reader = csv.DictReader(f)
			freq_list = []
			mass_list = []
			for row in reader:
				freq = float(row['Freq'])
				freq_list.append(freq)
				mass = float(row['Mass'])
				mass_list.append(mass)
		freq_list = [freq_list]
		mass_list = [mass_list]
		freq_list = np.array(freq_list)
		mass_list = np.array(mass_list)
		_, _, freq_knn = freq_model.predict(freq_list)
		_, _, mass_knn = mass_model.predict(mass_list)
		freq_knn = freq_knn[0]
		mass_knn = mass_knn[0]
		print("Freq:", freq_knn)
		print("Mass:", mass_knn)
		print(" ")
		freq_probability = {}
		freq_probability['plastic'] = int((freq_knn == 0).sum())/10
		freq_probability['metal'] = int((freq_knn == 1).sum())/10
		freq_probability['glass'] = int((freq_knn == 2).sum())/10
		#freq_probability['paper'] = int((freq_knn == 3).sum())/10
		mass_probability = {}
		mass_probability['plastic'] = int((mass_knn == 0).sum())/10
		mass_probability['metal'] = int((mass_knn == 1).sum())/10
		mass_probability['glass'] = int((mass_knn == 2).sum())/10
		#mass_probability['paper'] = int((mass_knn == 3).sum())/10
		print(freq_probability)
		print(mass_probability)
		end = time.time()
		process_time = end-start
		print("Time Taken:", process_time)
		print(" ")
		reply = str(freq_probability['plastic']) + " " + str(freq_probability['metal']) + " " + str(freq_probability['glass']) + " " + str(mass_probability['plastic']) + " " + str(mass_probability['metal']) + " " + str(mass_probability['glass']) + " - " + str(process_time)
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