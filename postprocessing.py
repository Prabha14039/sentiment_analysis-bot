import numpy as np
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences


def Postproccessing(df: str) -> np.array:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([df])

    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences([df])

    # Padding to make the sequence a uniform length (e.g., maxlen=10)
    max_sequence_length = 1  # Define a suitable length
    padded_sequence = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

    # The resulting padded_sequence is a 2D array; reshape it to a 3D array
    n_samples, n_time_steps = padded_sequence.shape
    n_features = max(tokenizer.word_index.values()) + 1  # +1 for the padding token

    # Reshape into a 3D array
    X = padded_sequence.reshape(n_samples, n_time_steps, 1)
    return X