
import streamlit as st
st.title('Movie review classification')
st.markdown("This project uses the IMDB movie review dataset")
st.markdown("Using sentiment analysis, reviews will be classified under a specific category")
st.sidebar.title("Steps for use: ")
st.sidebar.markdown("1. Write a review in the box")
st.sidebar.markdown("2. Press enter")
st.sidebar.markdown("3. Wait for Positive/Negative to appear")
st.sidebar.markdown("It's that simple!")

# Import modules
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
# For preprocessing reviews
import nltk 
# To use Regular expression for clean up of text
import re 
# Plotting a graph for dataset
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
# Save trained model to pickle 
import pickle
# Downloading stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords 
# Stemming of words
from nltk.stem.porter import PorterStemmer 

# Read the dataset
dataset = pd.read_table('moviereviews.tsv')
# Return the shape of data 
print("Shape:",dataset.shape)
print()
# Return the n dimensions of data
print("Dimensions:",dataset.ndim)  
print()
# Return the size of data  
print("Size:",dataset.size) 
print()
# Returns the sum fo all null values
print("Sum of NA characters:",dataset.isna().sum())  
print()
# Give concise summary of a DataFrame
print("Summary:",dataset.info())  
print()
# Display top 5 rows of the dataframe
print("Top 5 reviews:",dataset.head()) 
print()
# Display bottom 5 rows of the dataframe
print("Last 5 reviews:",dataset.tail())
print()
# Display how many positive and negative reviews are present
print(sns.countplot('sentiment',data=dataset))

corpus = []
# Total reviews = 25000
for i in range(0,25000):  
  # Replace , in review by a white space
  review = re.sub('[^a-zA-Z]'," ",dataset["review"][i])
  # Convert all text to lower case
  review = review.lower()
  # Separate all words
  review = review.split()
  pe = PorterStemmer()
  # Generate stop words (commonly occuring words)
  all_stopword = stopwords.words('english')
  all_stopword.remove('not')
  # Append all non-stop words to corpus list
  review = [pe.stem(word) for word in review if not word in set(all_stopword)]
  review = " ".join(review)
  corpus.append(review)

# Creating bag of word model and converting reviews to binary form
cv = CountVectorizer(max_features=1500) 
X = cv.fit_transform(corpus).toarray()
y = dataset["sentiment"]

# Split input data into testing and training sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Fit data on Gaussian and Multinomial models
GNB = GaussianNB()
MNB = MultinomialNB()
model1 = GNB.fit(X_train, y_train)
model2 = MNB.fit(X_train, y_train)

# Get score for both models and choose best one
print(GNB.score(X_test,y_test))   # 0.7554
print(MNB.score(X_test,y_test))   # 0.8462

# Predict output for test data
y_pred=model2.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), np.array(y_test).reshape(len(y_test),1)),1))

# Evaluate model
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test,y_pred)
cl_report = classification_report(y_test,y_pred)
print("Confusion matrix:\n",cm)
print("Classification report:\n",cl_report)
print("Accuracy of MNB: ",score*100)

# Save trained model and CountVectorizer to pickle
pickle.dump(cv, open('cv.pkl', 'wb'))
pickle.dump(model2, open("review.pkl", "wb"))
# Open trained model and re-evaluate
loaded_model = pickle.load(open("review.pkl", "rb"))
y_pred_new = loaded_model.predict(X_test)
print("Accuracy score: ",loaded_model.score(X_test,y_test))

# Predict output for new review
def new_review(new_review):
  new_review = new_review
  # Replace , by white spaces in the input review
  new_review = re.sub('[^a-zA-Z]', ' ', new_review)
  # Convert to lower case
  new_review = new_review.lower()
  # Separate words from text in review
  new_review = new_review.split()
  ps = PorterStemmer()
  # Generate stop words in english
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  # If word in review is not a stop word, add to corpus list
  new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
  new_review = ' '.join(new_review)
  new_corpus = [new_review]
  # Convert to binary form
  new_X_test = cv.transform(new_corpus).toarray()
  # Predict sentiment for review
  new_y_pred = loaded_model.predict(new_X_test)
  return new_y_pred

# Take review from user
input_review = st.text_input('Enter new review:')
new_review = new_review(input_review)
# Display results to user
if new_review[0]==1:
   st.title("Positive")
else :
   st.title("Negative")
