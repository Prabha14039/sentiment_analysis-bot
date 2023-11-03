import pandas as pd
from data_preperation import data_prep
from preprocessing import tokenize
from training import train, splitting
from models import model_train

df = pd.read_csv('IMDB Dataset.csv')
X, y = data_prep(df, 'review', 'sentiment')
X_train, x_test, y_train, y_test = train(X, y)

X_train, x_test, vocab_length = tokenize(X_train, x_test, 100)
X, Y, splits = splitting(X_train, x_test, y_train, y_test)

model_train(X, Y, splits)





