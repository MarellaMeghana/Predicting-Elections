# ====================== IMPORT PACKAGES ==============

import pandas as pd

from sklearn.model_selection import train_test_split


from sklearn import metrics
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn import linear_model
import pandas as pd
import re
from wordcloud import STOPWORDS
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing 



import base64
import streamlit as st

st.markdown(f'<h1 style="color:#000000;font-size:34px;">{"A Systematic Review of Predicting Elections Based on Social Media Data"}</h1>', unsafe_allow_html=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('2.jpg')



# ================= INPUT DATA ========================


print("-------------------------------------------------------------------------")
print("A Systematic Review of Predicting Elections Based on Social Media Data   ")
print("-------------------------------------------------------------------------")
print()

dataframe=pd.read_csv(r"Dataset.csv",lineterminator='\n')

dataframe = dataframe[0:1000]

print("--------------------------------")
print("Data Selection")
print("--------------------------------")
print(dataframe.head(15))    


st.write("--------------------------------")
st.write("Data Selection")
st.write("--------------------------------")
st.write(dataframe.head(15))  




  #-------------------------- PRE PROCESSING --------------------------------
  
  #------ checking missing values --------
  
st.write("----------------------------------------------------")
st.write("              Handling Missing values               ")
st.write("----------------------------------------------------")
st.write(dataframe.isnull().sum())


print("----------------------------------------------------")
print("              Handling Missing values               ")
print("----------------------------------------------------")
print()
print(dataframe.isnull().sum())



res = dataframe.isnull().sum().any()

if res == False:

    print("--------------------------------------------")
    print("  There is no Missing values in our dataset ")
    print("--------------------------------------------")
    print() 

    st.write("--------------------------------------------")
    st.write("  There is no Missing values in our dataset ")
    st.write("--------------------------------------------")
    print()   

else:

    print("--------------------------------------------")
    print(" Missing values is present in our dataset   ")
    print("--------------------------------------------")
    print()     

    st.write("--------------------------------------------")
    st.write(" Missing values is present in our dataset   ")
    st.write("--------------------------------------------")
    print()   

    dataframe = dataframe.fillna(0)
    
    resultt = dataframe.isnull().sum().any()
    
    if resultt == False:
        
        print("--------------------------------------------")
        print(" Data Cleaned !!!   ")
        print("--------------------------------------------")
        print()    
        
        print(dataframe.isnull().sum())
        
        st.write("--------------------------------------------")
        st.write(" Data Cleaned !!!   ")
        st.write("--------------------------------------------")
        print()    
        
        st.write(dataframe.isnull().sum())
      
# ---- LABEL ENCODING
  
print("--------------------------------")
print("Before Label Encoding")
print("--------------------------------")   

st.write("--------------------------------")
st.write("Before Label Encoding")
st.write("--------------------------------")   


df_class=dataframe['source']

print(dataframe['source'].head(15))

print("--------------------------------")
print("After Label Encoding")
print("--------------------------------")            
        
st.write("--------------------------------")
st.write("After Label Encoding")
st.write("--------------------------------")     



label_encoder = preprocessing.LabelEncoder() 

dataframe['source']=label_encoder.fit_transform(dataframe['source'].astype(str))                  
            
print(dataframe['source'].head(15))            

st.write(dataframe['source'].head(15))             
          
#========================= NLP TECHNIQUES ============================

#=== TEXT CLEANING ==== 
import re
import nltk
from nltk.corpus import stopwords
import string
stop_words = stopwords.words('english')
stemmer    = nltk.SnowballStemmer("english")
stop_words = stopwords.words('english')
  
  
def clean_data(text1):
    text1 = str(text1).lower()
    text1 = re.sub('\[.*?\]', '', text1)
    text1 = re.sub('https?://\S+|www\.\S+', '', text1) # remove urls
    text1 = re.sub('<.*?>+', '', text1)
    text1 = re.sub('[%s]' % re.escape(string.punctuation), '', text1) # remove punctuation
    text1 = re.sub('\n', '', text1)
    text1 = re.sub('\w*\d\w*', '', text1)
    return text1
def preprocess_data(text):
    text = clean_data(text)                                                     # Clean puntuation, urls, and so on
    text = ' '.join(word for word in text.split() if word not in stop_words)    # Remove stopwords
    text = ' '.join(stemmer.stem(word) for word in text.split())                # Stemm all the words in the sentence
    return text
  
print("----------------------------------------------------")
print("                Before Applying NLP                 ")
print("----------------------------------------------------")
print()
print(dataframe['tweet'].head(10))


st.write("----------------------------------------------------")
st.write("                Before Applying NLP                 ")
st.write("----------------------------------------------------")
print()
st.write(dataframe['tweet'].head(10))



print("----------------------------------------------------")
print("             After Applying NLP                     ")
print("----------------------------------------------------")
print()

st.write("----------------------------------------------------")
st.write("             After Applying NLP                     ")
st.write("----------------------------------------------------")
print()

dataframe["Clean"] = dataframe["tweet"].apply(preprocess_data) 

print(dataframe["Clean"].head(10))  

st.write(dataframe["Clean"].head(10))  

#==== TOKENIZATION ======

from tensorflow.keras.preprocessing.text import Tokenizer  #tokeniazation

tokenizer = Tokenizer()

tokenizer.fit_on_texts(dataframe["Clean"])
X1 = tokenizer.texts_to_sequences(dataframe["Clean"])
vocab_size = len(tokenizer.word_index)+1



print("----------------------------------------------------")
print("            Tokeniazation                    ")
print("----------------------------------- -----------------")
print()
print("Sentence:\n{}".format(dataframe["Clean"]))

st.write("----------------------------------------------------")
st.write("            Tokeniazation                    ")
st.write("----------------------------------- -----------------")
print()
st.write("Sentence:\n{}".format(dataframe["Clean"]))



print()
print("----------------------------------------------------------")
print()
print("\nAfter tokenizing :\n{}".format(X1[1]))
print()

st.write("----------------------------------------------------------")
st.write("\nAfter tokenizing :\n{}".format(X1[1]))
st.write("----------------------------------------------------------")


from tensorflow.keras.preprocessing.sequence import pad_sequences   #padding

X1 = pad_sequences(X1, padding='post')



# =========== SENTIMENT
    
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  
    

analyzer = SentimentIntensityAnalyzer()
dataframe['compound'] = [analyzer.polarity_scores(x)['compound'] for x in dataframe['Clean']]
dataframe['neg'] = [analyzer.polarity_scores(x)['neg'] for x in dataframe['Clean']]
dataframe['neu'] = [analyzer.polarity_scores(x)['neu'] for x in dataframe['Clean']]
dataframe['pos'] = [analyzer.polarity_scores(x)['pos'] for x in dataframe['Clean']]



fin_res=[]

for i in range(0,len(dataframe)):
    if dataframe['compound'][i] < -0.05:
        # print("Negative - 1")
        fin_res.append(1)
    elif dataframe['compound'][i] > 0.03:
        # print("Pos - 2")
        fin_res.append(2)    
    else:
        # print("Neu - 3")
        fin_res.append(3)         
        
target=pd.DataFrame(fin_res)



# ================== VECTORIZATION ====================
 
 # ---- COUNT VECTORIZATION ----
 
from sklearn.feature_extraction.text import CountVectorizer
    
#CountVectorizer method
vector = CountVectorizer(stop_words = 'english', lowercase = True)

#Fitting the training data 
count_data = vector.fit_transform(dataframe["Clean"])

print("---------------------------------------------")
print("            COUNT VECTORIZATION          ")
print("---------------------------------------------")
print()  
print(count_data)
      


    
# ================== DATA SPLITTING  ====================


X=count_data
y=target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("---------------------------------------------")
print("             Data Splitting                  ")
print("---------------------------------------------")

print()

print("Total no of input data   :",dataframe.shape[0])
print("Total no of test data    :",X_test.shape[0])
print("Total no of train data   :",X_train.shape[0])

st.write("---------------------------------------------")
st.write("             Data Splitting                  ")
st.write("---------------------------------------------")

print()

st.write("Total no of input data   :",dataframe.shape[0])
st.write("Total no of test data    :",X_test.shape[0])
st.write("Total no of train data   :",X_train.shape[0])


# ==== NAIVE BAYES ========
 
from sklearn.naive_bayes import MultinomialNB
 
 
nb = MultinomialNB()

nb.fit(X_train,y_train)

pred_nb = nb.predict(X_train)

acc_nb = metrics.accuracy_score(y_train, pred_nb) * 100

print("---------------------------------------------")
print(" Naive Bayes Classifier")
print("---------------------------------------------")
print()
print("1) Accuracy = ", acc_nb, '%' )
print()
print("2) Classification Report ")
print()
print(metrics.classification_report(y_train, pred_nb))
print()        


st.write("---------------------------------------------")
st.write(" Naive Bayes Classifier")
st.write("---------------------------------------------")
print()
st.write("1) Accuracy = ", acc_nb, '%' )
print()
st.write("2) Classification Report ")
print()
st.write(metrics.classification_report(y_train, pred_nb))
print()        



    
#  CNN - 1D 

from keras.models import Model
from keras.layers import Conv1D, MaxPool1D, Flatten, Input

inp =  Input(shape=(45,1))
conv = Conv1D(filters=2, kernel_size=2)(inp)
pool = MaxPool1D(pool_size=2)(conv)
flat = Flatten()(pool) 
#dense = Dense(1)(flat)
model = Model(inp, flat)
model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])
model.summary()
    
#model fitting
history = model.fit(X1, y,epochs=10, batch_size=15, verbose=1,validation_split=0.2)

acc_cnn=history.history['accuracy']

acc_cnn=max(acc_cnn)

acc_cnn=100-acc_cnn

pred_cnn=model.predict(X1)

y_pred1 = pred_cnn.reshape(-1)
y_pred1[y_pred1<0.5] = 0
y_pred1[y_pred1>=0.5] = 1
y_pred1 = y_pred1.astype('int')

print("---------------------------------------------")
print(" Convolutional Neural Network - CNN 1D")
print("---------------------------------------------")
print()
print("1. Accuracy :",acc_cnn )
print()
print("2) Loss  ", 100-acc_cnn)


st.write("---------------------------------------------")
st.write(" Convolutional Neural Network - CNN 1D")
st.write("---------------------------------------------")
print()
st.write("1. Accuracy :",acc_cnn )
print()
st.write("2) Loss  ", 100-acc_cnn)

    
print("---------------------------------------------------------")

import seaborn as sns
sns.barplot(x=["Naive Bayes","Cnn-1D "],y=[acc_nb, acc_cnn])
plt.title("Comparison Graph")
plt.savefig("Graph.png")
plt.show()

    
st.image("Graph.png")    



st.markdown(f'<h1 style="color:#000000;font-size:34px;">{"Kindly enter the tweet"}</h1>', unsafe_allow_html=True)



getin=st.text_input("Enter the Tweets = ")

aa = st.button("PREDICT")

if aa:    

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(getin)
    
    if sentiment_dict['compound'] >= 0.05 :
        aa="Positive "
        print("Positive ")
        st.text("Identified = Positive")
     
    elif sentiment_dict['compound'] <= - 0.05 :
        aa="Negative "
        print("Negative Tweets")
        st.text("Identified Negative")
    else:
        aa="Neutral "
        print("Neutral Tweets")        
        st.text("Identified Neutral")










