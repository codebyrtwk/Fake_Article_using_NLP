  
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



# DATA PREPROCESSING using kpg_talkies
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
	note = st.write("Note : This app will supposed to do prediction on the basis of Article's Features (TF-IDF) using SGD CLassifer!!")
	article = st.text_area('Enter the News Article')
	ml_models = ["SGD" , "Random Forest" , "KNN"]
	models_select = st.selectbox("Please Choose a Model" , ml_models)
	submit_button = st.button('Predict')

	
	if submit_button and len(article)==0 :
		st.markdown("Sorry , You haven't entered any article yet !!!!!")
		


		
	if submit_button and len(article) != 0 :
		
		#INPUT TEXT PREPROCESSING
		article_processed = get_clean(article)

		#tokenizer
		tokens = word_tokenize(article_processed)

		#lemmatizing tokens
		lemmatizer = nltk.stem.WordNetLemmatizer()	
		lemmatized_tokens = [lemmatizer.lemmatize(t) for t in tokens]		


		#Using vectorizer for feature extraction
		tfidf = TfidfVectorizer()  
		vector = tfidf.fit_transform(lemmatized_tokens)
		vector = vector.todense()
		vector = np.resize(vector, (1, 106186))

		

		prediction = clf.predict(vector)
		st.write("Prediction : " , str(prediction))
		




# Text Preprocessing Page
if choice == "Text Preprocessing":
	st.title("Pre Process the text")
	article = st.text_area('Enter the text data')
	select_processing = ["Process" , "Tokenize" , "Vectorizer"]
	choice1 = st.selectbox("Choose" ,select_processing )
	
	submit_button = st.button("Submit")
	if choice1 == "Process" and submit_button :
		article_processed = get_clean(article)
		st.markdown( article_processed)

	if  choice1 == "Tokenize" and submit_button :
		article_processed = get_clean(article)
		tokens = word_tokenize(article_processed)
		st.markdown(tokens)

	if choice1 == "Vectorizer" and submit_button:
		article_processed = get_clean(article)
		tokens = word_tokenize(article_processed)
		tfidf = TfidfVectorizer()
		vector = tfidf.fit_transform(tokens)
		st.markdown(vector)

	for i in range(len(select_processing)):
		if submit_button and len(article) == 0 :
			if (select_processing[i]):
				st.write("You haven't entered any text yet!!")
				break

