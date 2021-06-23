import streamlit as st
import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocess_kgptalkie as ps
import re
import parse
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn import *
import cv2



#importing pickle
sgd = open('SGD_.pkl','rb')
clf = joblib.load(sgd)


activities = ['Prediction' , "Text Preprocessing" , 'Confusion Matrix' , 'WordCloud']
choice = st.sidebar.selectbox("Choose" , activities)



# DATA PREPROCESSING
def get_clean(x):
    x = str(x).lower().replace('\\', '').replace('_', ' ')
    x = ps.cont_exp(x)
    x = ps.remove_emails(x)
    x = ps.remove_urls(x)
    x = ps.remove_rt(x)
    x = ps.remove_accented_chars(x)
    x = ps.remove_special_chars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x

	
    	

# Prediction Page

if choice == "Prediction":
	heading = st.title("Fake News Classifier")
	note = st.info("Note : This app will supposed to do prediction on the basis of Article's Features (TF-IDF) using SGD CLassifer!!")
	article = st.text_area('Enter the News Article')
	submit_button = st.button('Predict')

	
	if submit_button and len(article)==0 :
		st.info("Sorry , You haven't entered any article yet !!!!!")
		


		
	if submit_button and len(article) != 0 :
		st.info( article)
		#INPUT TEXT PREPROCESSING
		article_processed =get_clean(article)

		#tokenizer
		tokens = word_tokenize(article_processed)

		#Using vectorizer for feature extraction
		tfidf = TfidfVectorizer()  
		vector = tfidf.fit_transform(tokens)
		vector = vector.todense()
		vector = np.resize(vector, (1, 106186))

		
		st.write("Prediction :")

		prediction = clf.predict(vector)
		st.info(str(prediction))
		




		
#confusion matrix

if choice == "Confusion Matrix":
	st.title("Confusion matrix")
	def img_to_bytes(img_path):
		img_bytes = Path(img_path).read_bytes()
		encoded = base64.b64encode(img_bytes).decode()
		return encoded
	image1 = img_to_bytes("SGD.png")
	st.image(image1)
    

if choice == "Text Preprocessing":
	
	st.title("Pre Process the text")
	article = st.text_area('Enter the text data')
	select_processing = ["Process" , "Tokenize" , "Vectorizer"]
	choice1 = st.selectbox("Choose" ,select_processing )
	

	if choice1 == "Process" and st.button('Submit') :
		article_processed = get_clean(article)
		st.info(article_processed)

	if  choice1 == "Tokenize" and st.button('Submit') :
		article_processed = get_clean(article)
		tokens = word_tokenize(article_processed)
		st.info(tokens)

	if choice1 == "Vectorizer" and st.button('Submit'):
		article_processed = get_clean(article)
		tokens = word_tokenize(article_processed)
		tfidf = TfidfVectorizer()
		vector = tfidf.fit_transform(tokens)
		st.info(vector)



		


	
	






