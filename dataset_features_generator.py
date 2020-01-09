
import pickle
import pandas as pd

from tqdm import tqdm

#For DELF loading
import tensorflow as tf
import tensorflow_hub as hub

from utils import *

def generate_dataset_vectors(paths):
	'''
	Call this method to generate feature vectors for each image in the dataset.
	'''
	
	print(len(paths))
	tf.reset_default_graph()
	tf.logging.set_verbosity(tf.logging.FATAL)

	model = hub.Module('https://tfhub.dev/google/delf/1')

	image_placeholder = tf.placeholder(tf.float32, shape=(None, None, 3), name='input_image')

	module_inputs = {
		'image': image_placeholder,
		'score_threshold': 100.0,
		'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
		'max_feature_num': 1000,
	}

	module_outputs = model(module_inputs, as_dict=True)

	image_tf = paths_to_image_loader(list(paths))

	with tf.train.MonitoredSession() as sess:
		results_dict = {}
		for i in tqdm(range(len(paths))):
			image_path = paths[i]
			image = sess.run(image_tf)
			results_dict[image_path] = sess.run([module_outputs['locations'], module_outputs['descriptors']],feed_dict={image_placeholder: image})
			if i % 1000 == 0 and i >0:
				print("Saving sec reasons -> ", i)
				with open("patike_security_data_features.pickle".format(i), 'wb') as f:
					pickle.dump(results_dict, f)

	
	return results_dict


final_dataset = pd.read_csv("dataset_builder_module/dataset/main_dataset.csv")
rea_dict = generate_dataset_vectors(final_dataset.iloc[:, -1].values)
with open("patike_features.pickle", 'wb') as f:
	pickle.dump(rea_dict, f)