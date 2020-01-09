import os
import cv2
import json
import time
import pickle
import werkzeug
import pandas as pd

#For DELF loading
import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image, ImageOps
from utils import find_close_books

#import Flask dependencies
from flask import Flask, request, render_template, send_from_directory, jsonify

#Set root dir
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

with open("dataset_features.pickle", 'rb') as f:
	dataset_features = pickle.load(f)

dataset = pd.read_csv("dataset_builder_module/dataset/main_dataset.csv")


#Define Flask app
app = Flask(__name__, static_url_path='/static')

#Define apps home page
@app.route("/") #www.image-search.com/
def index():
	return render_template("index.html")

#Define upload function
@app.route("/api/android-upload", methods=["POST"])
def upload():

	upload_dir = os.path.join(APP_ROOT, "uploads/")

	if not os.path.isdir(upload_dir):
		os.mkdir(upload_dir)

	imagefile = request.files['image']
	filename = werkzeug.utils.secure_filename(imagefile.filename)
	imagefile.save(upload_dir + filename)

	#Perform the inference process on the uploaded image
	result = find_close_books(upload_dir + filename, 
							upload_dir + filename, 
							dataset_features, 
							dataset.iloc[:len(dataset_features), -1].values,)

	results = pd.DataFrame(result)['index'].values
	data = dataset.iloc[:len(dataset_features)].values[results]
	
	#Create resulting DataFrame object
	result =  {}
	columns = ['image', 'name', 'author', 'format', 'book_depository_stars', 'price', 'currency', 'old_price' ,'isbn', 'category']
				
	for i in range(len(data)):
		obj = {}
		for k in range(len(columns)):
			obj[columns[k]] = data[i][k]

		result[i] = obj
	
	return json.dumps(result)

#Define helper function for finding image paths
@app.route("/upload/<filename>")
def send_image(filename):
	return send_from_directory("uploads", filename)

#Start the application

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)