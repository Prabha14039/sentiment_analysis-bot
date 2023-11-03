from flask import Flask, render_template, request
from tsai.inference import load_learner
from postprocessing import Postproccessing

app = Flask(__name__)

# @-->is used to represent a decorator
# decorator --> modifies the behavior of a function or a class
# /--> indicates the homepage
@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    Statement = str(request.form['statement'])
    X = Postproccessing(Statement)
    fcst = load_learner("models/fcst1.pkl", cpu=False)
    raw_preds, target, preds = fcst.get_X_preds(X)
    return render_template('index.html',prediction_text =f'Sentiment {preds},{target},{raw_preds}')

if __name__ == "__main__":
    app.run()