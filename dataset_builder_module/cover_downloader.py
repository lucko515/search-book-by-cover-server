import os
import urllib
import pandas as pd

from tqdm import tqdm


def cover_downloader(folder, csv_file):
	'''
	Call this function to download all covers for a single category.

	folder - String, folder name for the subcategory
	csv_file - String, path to the csv_file for the current category
	'''

	csv = pd.read_csv(csv_file)
	
	#Extract URLs from the csv file
	images = csv['image'].values
	
	all_images = []
	print("Working on: ", csv_file)
	for i in tqdm(range(1, len(images)+1)):
		try:
			#Create the path to the image
			img_name = folder+"/{:07d}.jpg".format(i)
			f = open(img_name, 'wb')
			#Download and save the current imgs
			f.write(urllib.request.urlopen(images[i-1]).read())
			f.close()
			all_images.append(img_name)
		except Exception as e:
			print(e)
			all_images.append("")
		
	assert len(csv) == len(all_images)
	csv['img_paths'] = all_images
	csv.to_csv(csv_file, index=False)


def download_all(top_dataset_dir, make_new_top_file=True):
	'''
	The main downloader function.

	top_dataset_dir - String, path to the folder where the whole dataset is located
	make_new_top_file - Boolean, if True, this function will create a new csv file that contains local paths to images
	'''

	subfolders = [f.path for f in os.scandir(top_dataset_dir) if f.is_dir() ]
	csv_to_concat = []
	for folder in subfolders:
		csv_file = folder + "/" + list(filter(lambda x: x.endswith('.csv'), os.listdir(folder)))[0]
		cover_downloader(folder, csv_file)
		csv_to_concat.append(csv_file)
		
	if make_new_top_file:
		all_csvs = []
		for csv in csv_to_concat:
			all_csvs.append(pd.read_csv(csv))
		
		top_file = pd.concat(all_csvs)
		top_file.to_csv(top_dataset_dir + "/main_dataset.csv", index=False)
		print("Dataset made.")


if __name__ == "__main__":
	download_all("dataset/")