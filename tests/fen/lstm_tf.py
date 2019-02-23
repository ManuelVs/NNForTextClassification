import pickle
import numpy as np
import sys
from fen.classifier import Classifier
from fen.lstm_tf import LSTMModel

if __name__ == '__main__':
    filePathEncodedTREC = sys.argv[1]
    filePathWord2vec = sys.argv[2]

    with open(filePathEncodedTREC, "rb") as f:
        (X_train, y_train), (X_eval, y_eval) = pickle.load(f)
    with open(filePathWord2vec, 'rb') as file:
        word_vector = np.load(file)
    
    word_vector = word_vector.astype('float32')
    sentence_length = X_train[0].shape[0]
    num_classes     = y_train[0].shape[0]

    model = LSTMModel(
        sentence_length=sentence_length,
        embedding=word_vector,
        num_classes=num_classes
    )

    classifier = Classifier(model)

    classifier.train(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        epochs=20,
        batch_size=32
    )
