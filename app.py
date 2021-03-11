# Import modules
import pandas as pd 
import pickle
import streamlit as st
import re
import sklearn

# Display on webpage
st.title("Movie review classification")
st.markdown("This project uses the IMDB movie review dataset")
st.markdown("Using sentiment analysis, reviews will be classified under a specific category")
st.sidebar.title("Steps for use: ")
st.sidebar.markdown("1. Write a review in the box")
st.sidebar.markdown("2. Press enter")
st.sidebar.markdown("3. Wait for Positive/Negative to appear")
st.sidebar.markdown("It's that simple!")

# Load previously created models
loaded_model = pickle.load(open("MNBmodel.pkl","rb"))
cv = pickle.load(open("countvectorizer.pkl","rb"))

# Predict output for new review
def new_review(new_review):
  new_review = new_review
  # Replace , by white spaces in the input review
  new_review = re.sub('[^a-zA-Z]', ' ', new_review)
  # Convert to lower case
  new_review = new_review.lower()
  # Separate words from text in review
  new_review = new_review.split()
  # Generate stop words in english
  all_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
  # If word in review is not a stop word, add to corpus list
  new_review = [word for word in new_review if not word in set(all_stopwords)]
  new_review = ' '.join(new_review)
  new_corpus = [new_review]
  # Convert to binary form
  new_X_test = cv.transform(new_corpus).toarray()
  # Predict sentiment for review
  new_y_pred = loaded_model.predict(new_X_test)
  return new_y_pred

# Take review from user
input_review = st.text_input("Enter new review:")
new_review = new_review(input_review)
# Display results to user
if new_review[0]==1:
   st.title("Positive")
else :
   st.title("Negative")
