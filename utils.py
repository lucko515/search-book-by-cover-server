
import os
import pickle
import numpy as np
import pandas as pd
#For DELF loading
import tensorflow as tf
import tensorflow_hub as hub

#Image dependencies
from PIL import Image, ImageOps

#Compare resulting images in the speed of log(n)
from scipy.spatial import cKDTree

#Remove bad results with Random Sampling
from skimage.measure import ransac
from skimage.transform import AffineTransform

#For Inlines calculation
from itertools import accumulate


#Create the TensorFlow graph object
g = tf.Graph()

#Load the DELF model from the TF Hub
with g.as_default():
    model = hub.Module('https://tfhub.dev/google/delf/1') 

    #Define image placeholder that supports RGB images of any sizes
    image_placeholder = tf.placeholder(
                    tf.float32, shape=(None, None, 3), name='input_image')

    module_inputs = {
                    'image': image_placeholder,
                    'score_threshold': 100.0,
                    'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0], #these scales are taken form DELF paper
                    'max_feature_num': 1000,
                }

    module_outputs = model(module_inputs, as_dict=True)


def paths_to_image_loader(image_files):
    '''
    This functions loads one by one image from the folder.
    '''
    #Creates Queue from the list of paths
    filename_queue = tf.train.string_input_producer(image_files, shuffle=False)
    
    #Create file reader
    reader = tf.WholeFileReader()
    
    #Read queued files
    _, value = reader.read(filename_queue)
    
    #Conver loaded binary to the JPEG image in the RGB format
    image_tf = tf.image.decode_jpeg(value, channels=3)
    
    #Cast pixesl from int to float32 and return it
    return tf.image.convert_image_dtype(image_tf, tf.float32)


def resize_image(srcfile, destfile, new_width=128, new_height=128):
    '''
    Helper function that resizes an image to a specified new width and height.

    srcfile - String, path to the original image
    destfile - String, path where the resized image will be saved
    new_width - Integer
    new_height - Integer
    '''
    
    #Opens the original image
    pil_image = Image.open(srcfile)
    
    #resizes the original image and convers it back to RGB format
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert('RGB')
    
    #Saves the resized image to the disk
    pil_image_rgb.save(destfile, format='JPEG', quality=90)
    return destfile


def query_image_features_generator(image_path):
    '''
    This function takes newly uploaded image (Query image) and generates features with the pre-trained model for that image.
    '''
    with tf.Session(graph=g) as s:
      
        #prepairs the image for the model
        image_tf = paths_to_image_loader([image_path])

        with tf.train.MonitoredSession() as sess:
            image = sess.run(image_tf)
            print('Extracting locations and descriptors from %s' % image_path)
            #Generate features
            return sess.run([module_outputs['locations'], module_outputs['descriptors']],feed_dict={image_placeholder: image})


def image_index_2_accumulated_indexes(index, accumulated_indexes_boundaries):
    '''
    Image index to accumulated/aggregated locations/descriptors pair indexes.
    '''
    if index > len(accumulated_indexes_boundaries) - 1:
        return None
    accumulated_index_start = None
    accumulated_index_end = None
    if index == 0:
        accumulated_index_start = 0
        accumulated_index_end = accumulated_indexes_boundaries[index]
    else:
        accumulated_index_start = accumulated_indexes_boundaries[index-1]
        accumulated_index_end = accumulated_indexes_boundaries[index]
    return np.arange(accumulated_index_start,accumulated_index_end)


def get_locations_2_use(image_db_index, 
                        k_nearest_indices, 
                        accumulated_indexes_boundaries,
                        query_image_locations, 
                        locations_agg):
    '''
    Get a pair of locations to use, the query image to the database image with given index.
    Return: a tuple of 2 numpy arrays, the locations pair.
    '''
    image_accumulated_indexes = image_index_2_accumulated_indexes(image_db_index, accumulated_indexes_boundaries)
    locations_2_use_query = []
    locations_2_use_db = []
    for i, row in enumerate(k_nearest_indices):
        for acc_index in row:
            if acc_index in image_accumulated_indexes:
                locations_2_use_query.append(query_image_locations[i])
                locations_2_use_db.append(locations_agg[acc_index])
                break

    return np.array(locations_2_use_query), np.array(locations_2_use_db)


def query_image_pipeline(image_path, save_path):
    '''
    Model inference wrapper. This function queries the model with the Query image and generates its features.

    image_path - String, path to the Query image
    save_path - String, path where the Query image is saved
    '''
    resized_image = resize_image(image_path, save_path)
    
    query_image_locations, query_image_descriptors = query_image_features_generator(resized_image,)
    
    return query_image_locations, query_image_descriptors, resized_image


def find_close_books(image_path, 
                     save_path,
                     image_database,
                     paths,
                     distance_threshold = 0.8, 
                     k_neighbors=10, 
                     top_n=10):

    '''
    The inference function. Call this function to get top n close images based on the query image;

    image_path - String, path to the Query image
    save_path - String, path where the Query image is saved
    image_database - features generated based on the dataset
    paths - Numpy array, paths to database images (used to display closesest images)
    distance_threshold - Float, How similar two nodes should be in the KD tree
    k_neighbors -  Integer, How many neighbors is looked at in the KD tree
    top_n - Integer, How many similar images is retrieved
    '''

    query_image_locations, query_image_descriptors, resized_image = query_image_pipeline(image_path,
                                                                       save_path,
                                                                       )

    #TODO: This section slows down the whole inference process and should be moved outside of the function
    locations_agg = np.concatenate([image_database[img][0] for img in paths])
    descriptors_agg = np.concatenate([image_database[img][1] for img in paths])
    accumulated_indexes_boundaries = list(accumulate([image_database[img][0].shape[0] for img in paths]))
    
    #Create KD Tree
    d_tree = cKDTree(descriptors_agg)
    
    #Query the KD tree
    distances, indices = d_tree.query(query_image_descriptors,
                                      distance_upper_bound=distance_threshold, 
                                      k = k_neighbors,
                                      n_jobs=-1)


    # Find the list of unique accumulated/aggregated indexes
    unique_indices = np.array(list(set(indices.flatten())))

    unique_indices.sort()
    if unique_indices[-1] == descriptors_agg.shape[0]:
        unique_indices = unique_indices[:-1]
        
    
    unique_image_indexes = np.array(list(set([np.argmax([np.array(accumulated_indexes_boundaries)>index]) for index in unique_indices])))
    
    inliers_counts = []

    for index in unique_image_indexes:
        locations_2_use_query, locations_2_use_db = get_locations_2_use(index, indices, accumulated_indexes_boundaries, query_image_locations, locations_agg)

        # Perform geometric verification using RANSAC.
        try:
            _, inliers = ransac((locations_2_use_db, locations_2_use_query), # source and destination coordinates
                                AffineTransform,
                                min_samples=3,
                                residual_threshold=20,
                                max_trials=1000)
            
            # If no inlier is found for a database candidate image, we continue on to the next one.
            if inliers is None or len(inliers) == 0:
                continue
        except:
            continue
            
        # the number of inliers as the score for retrieved images.
        inliers_counts.append({"index": index, "inliers": sum(inliers)})
        
    
    result = sorted(inliers_counts, key=lambda k: k['inliers'], reverse=True)[:top_n]
    return result