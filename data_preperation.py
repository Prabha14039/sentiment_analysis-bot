from preprocessing import preprocess_text
import numpy as np


def data_prep(df, feature1: str, feature2: str):
    X = []
    sentences = list(df[feature1])
    for sen in sentences:
        X.append(preprocess_text(sen))

    Y = df[feature2]
    Y = np.array(list(map(lambda x: 1 if x == "positive" else 0, Y)))
    return X, Y
