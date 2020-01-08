import os
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from currency_converter import CurrencyConverter

from bookdepository_categories import links


def book_object_maker(book_source, converter, top_link):
	try:
		book_json = {}
		book_json['image'] = book_source.find("div", {"class":"item-img"}).img.get("data-lazy")
		book_json['name'] = book_source.find("div", {"class":"item-info"}).h3.text.strip()
		book_json['author'] = book_source.find("div", {"class":"item-info"}).find("p", {'class':"author"}).text.strip()
		book_json['format'] = book_source.find("div", {"class":"item-info"}).find("p", {'class':"format"}).text.strip()
		
		number_of_starts = 0
		for span in book_source.find("div", {"class":"item-info"}).find("div", {'class':"rating-wrap"}).findAll("span"):
			classes = span.get("class")
			if "full-star" in classes:
				number_of_starts += 1
			elif "half-star" in classes:
				number_of_starts += 0.5
				
		book_json['book_depository_stars'] = number_of_starts
		book_price_info = book_source.find("div", {"class":"item-info"}).find("div", {'class':"price-wrap"})
		book_price_info = book_price_info.p.text.split()
		if len(book_price_info) == 2:
			book_json['price'] = str(book_price_info[0].replace(",", "."))
			book_json['currency'] = "$"
			book_json['old_price'] = ""
		else:
			book_json['price'] = str(book_price_info[0].replace(",", "."))
			book_json['currency'] = "$"
			book_json['old_price'] = str(book_price_info[2].replace(",", "."))
			
		book_json['isbn'] = book_source.meta.get('content')
		book_json['category'] = top_link.split("/")[5]
		
		book_frame = pd.DataFrame(columns=["image", "name", "author", "format", "book_depository_stars", "price", "currency", "old_price", "isbn", "category"])
		bk = book_frame.append(book_json, ignore_index=True)
		return bk
	except Exception as e:
		print(e)
		return None


def category_scraper(page_link, converter, max_page=333):
	
	category_books = []
	
	for p in tqdm(range(1, max_page+1)):
		full_page_link = page_link + str(p)
		time.sleep(2)
		source = requests.get(full_page_link)
		#Convert to BS_object
		bs_obj = BeautifulSoup(source.text, 'lxml')
		products = bs_obj.find('div', {"class":"content-block"}).findAll("div", {"class":"book-item"})

		for product in tqdm(products):
			book_object = book_object_maker(product, 
										  converter, 
										  full_page_link)
			if book_object is not None:
				category_books.append(book_object)
			
	return pd.concat(category_books)


def books_scraper(dataset_dir, max_pages=333):
	
	all_books = []
	converter = CurrencyConverter()
	
	for link in links:
		print("Scraping: ", link)
		category = link.split("/")[5]
		category_folder = dataset_dir + category

		if not os.path.exists(category_folder):
			os.mkdir(category_folder)
		category_csv = category_scraper(link, converter=converter, max_page=max_pages)
		category_csv.to_csv(category + ".csv", index=False)
		all_books.append(category_csv)
		
	complete_csv = pd.concat(all_books)
	complete_csv.to_csv(dataset_dir + "dataset.csv", index=False)

if __name__ == "__main__":
	books_scraper("dataset/", max_pages=33)