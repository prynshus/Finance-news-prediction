from flask import Flask,render_template,url_for,request
from nltk.stem import WordNetLemmatizer
import pickle
import nltk
from nltk.corpus import stopwords

# Ensure 'punkt' is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Ensure 'stopwords' is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# If using lemmatizer
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

gb_clf = pickle.load(open('model.pkl', 'rb'))
vectorizer =pickle.load(open('transform.pkl', 'rb'))

app = Flask(__name__)
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        Lemmatizer = WordNetLemmatizer()
        words = nltk.word_tokenize(message)
        words = [Lemmatizer.lemmatize(word, pos='v') for word in words if word not in set(stopwords.words('english'))]
        message = ' '.join(words)
        vect = vectorizer.transform([message]).toarray()
        my_prediction = gb_clf.predict(vect)
    return render_template('results.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
