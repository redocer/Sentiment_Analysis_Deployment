from flask import Flask, render_template, request
import requests
import pickle
import numpy as np
import sklearn
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
lm = WordNetLemmatizer()
app = Flask(__name__)
model = pickle.load(open('sentiment_analysis_model.pkl', 'rb'))

def text_transformation(input_data):
    corpus = []
    for item in input_data:
        new_item = re.sub('[^a-zA-Z]',' ',str(item))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus
    

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Statement = [str(request.form['Statement'])]
        Processed_data = text_transformation(Statement)
        prediction=model.predict(Processed_data)
        output = int(prediction[0])
        if output == 0:
            return render_template('index.html',prediction_text="This sentence has a Negative Sentiment.")
        elif output == 1:
            return render_template('index.html',prediction_text="This sentence has a Positive Sentiment.")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

