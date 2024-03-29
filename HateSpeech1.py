import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import nltk
import re
from nltk.corpus import stopwords
stopwords=set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer('english')

data=pd.read_csv("cyberbullying_tweets.csv")
print(data.head())

data["labels"]=data["class"].map({0:"Hate Speech", 1:"Offensive Speech", 2:"No Hate and Offensive Speech"})
data=data[["tweet","labels"]]
data.head()

def clean (text):
    text = str (text). lower()
    text = re. sub('[.?]', '', text)
    text = re. sub('https?://\S+|www.\S+', '', text)
    text = re. sub('<.?>+', '', text)
    text = re. sub('[%s]' % re. escape(string. punctuation), '', text)
    text = re. sub('\n', '', text)
    text = re. sub('\w\d\w', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ". join(text)
    text = [stemmer. stem(word) for word in text. split(' ')]
    text=" ". join(text)
    return text
data["tweet"] = data["tweet"]. apply(clean)

x=np.array(data["tweet"])
y=np.array(data["labels"])

cv=CountVectorizer()

X = cv.fit_transform(x)

X_train,X_text,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train,y_train)

y_pred=model.predict(X_text)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

i="You are too bad and I dont like your attitude"
i = cv.transform([i]).toarray()
print(model.predict((i)))