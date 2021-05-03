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
	'''
	HTML to Python dict parser. Call this function to extrach book information from the HTML code.
	NOTE: It was custom created for Book Depository source code, so it only works for the BD website.

	:params:
	book_source - BeautifulSoup object, HTML source code of a book
	converter - CurrencyConverter object, Price currency converter, NOTE: For now it's not used
	top_link - String, URL of the current category, used to extract book category

	:return:
	DataFrame object with books' information
	'''
	try:
		#Extract basic books' information from the HTML code
		book_json = {}
		book_json['image'] = book_source.find("div", {"class":"item-img"}).img.get("data-lazy")
		book_json['name'] = book_source.find("div", {"class":"item-info"}).h3.text.strip()
		book_json['author'] = book_source.find("div", {"class":"item-info"}).find("p", {'class':"author"}).text.strip()
		book_json['format'] = book_source.find("div", {"class":"item-info"}).find("p", {'class':"format"}).text.strip()
		book_json['publication_date'] = book_source.find("div", {"class":"item-info"}).find("p", {'class':"published"}).text.strip()
		
		#Counts the books' rating
		number_of_starts = 0
		for span in book_source.find("div", {"class":"item-info"}).find("div", {'class':"rating-wrap"}).findAll("span"):
			classes = span.get("class")
			if "full-star" in classes:
				number_of_starts += 1
			elif "half-star" in classes:
				number_of_starts += 0.5
				
		book_json['book_depository_stars'] = number_of_starts

		#Extract books' price
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
		
		#Create DataFrame with books' info
		book_frame = pd.DataFrame(columns=["image", "name", "author", "format", "publication_date", "book_depository_stars", "price", "currency", "old_price", "isbn", "category"])
		bk = book_frame.append(book_json, ignore_index=True)
		return bk
	except Exception as e:
		print(e)
		return None


def category_scraper(page_link, converter, max_page=333):
	'''
	Call this function to scrape all books for a single category from the Book Depository website.

	:params:
	top_link - String, URL of the current category
	converter - CurrencyConverter object, Price currency converter, NOTE: For now it's not used
	max_page - Integer, Number of pages scraped for a category NOTE: After page 333 Book Depository is not showing more books, so the default value here is 333

	:return:
	DataFrame object with all books for the scraped category
	'''
	category_books = []
	
	#Iterate through all pages for the current category
	for p in tqdm(range(1, max_page+1)):
		full_page_link = page_link + str(p)
		time.sleep(2) #This sleep timer is used so the IP don't get block
		source = requests.get(full_page_link)
		#Convert to BS_object
		bs_obj = BeautifulSoup(source.text, 'lxml')
		#Extract all products from the current page
		products = bs_obj.find('div', {"class":"content-block"}).findAll("div", {"class":"book-item"})

		#Iterate through all products on the page
		for product in tqdm(products):
			book_object = book_object_maker(product, 
										  converter, 
										  full_page_link)
			if book_object is not None:
				category_books.append(book_object)
			
	return pd.concat(category_books)


def books_scraper(dataset_dir, max_pages=333):
	'''
	Top Book Depository scraper function. Call this function to start the whole scraping process.

	:params:
	dataset_dir - String, Path to the folder where all data will be saved
	max_page - Integer, Number of pages scraped for a category NOTE: After page 333 Book Depository is not showing more books, so the default value here is 333
	'''
	
	all_books = []
	converter = CurrencyConverter()
	
	#Iterate through all links found in the bookdepository_categories.py file
	for link in links:
		print("Scraping: ", link)
		category = link.split("/")[5]
		category_folder = dataset_dir + category

		#Create a sub folder for the category (if it does not exists)
		if not os.path.exists(category_folder):
			os.mkdir(category_folder)
		
		#Begin the scraping process for the current category
		category_csv = category_scraper(link, converter=converter, max_page=max_pages)
		category_csv.to_csv(category + ".csv", index=False)
		all_books.append(category_csv)
		
	complete_csv = pd.concat(all_books)
	complete_csv.to_csv(dataset_dir + "dataset.csv", index=False)

if __name__ == "__main__":
	books_scraper("dataset/", max_pages=333)