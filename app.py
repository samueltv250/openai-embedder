from flask import Flask, render_template, request
from qa_bot import *
from embedd import EmbeddingDataFrame
app = Flask(__name__)
embeddedDF = EmbeddingDataFrame()

embeddedDF.load_df("embeddedDF.pickle")

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = ""
    link = ""
    if request.method == 'POST':
        question = request.form['question']
        answer,link = answer_query_with_context(str(question), embeddedDF)
        # answer = 'hola'

    return render_template('index.html', answer=answer, link=link)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4040)
