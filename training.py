from sklearn.model_selection import train_test_split
from tsai.basics import *

def train(X, Y):
    X_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, x_test, y_train, y_test


def splitting(X_train, x_test, y_train, y_test):
    X, Y, splits = combine_split_data([X_train, x_test], [y_train, y_test])
    return X, Y, splits