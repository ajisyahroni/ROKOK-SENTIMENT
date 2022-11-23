from waitress import serve
from flask import request
import pandas as pd
from flask import Flask, render_template
import pickle



tfidf = pickle.load(open('models/tfidf.pkl', 'rb'))
bow = pickle.load(open('models/bow.pkl', 'rb'))
nb = pickle.load(open('models/nb.pkl', 'rb'))


app = Flask(__name__)
df = pd.read_csv('data/dataset.csv', sep=';')
df_pos = df[df['Value'] == 'Positif']
df_neg = df[df['Value'] == 'Negatif']


@app.route("/", methods=['GET', 'POST'])
def hello():
    if request.method == 'GET':
        return render_template('simple.html',  tables=[df.to_html(classes='table table-stripped')], titles=df.columns.values)
    if request.method == 'POST':
        contoh = [request.form.get("tweet-input")]
        jk = tfidf.transform(bow.transform(contoh))
        return nb.predict(jk)[0]


if __name__ == "__main__":
    serve(app, port=8080)
