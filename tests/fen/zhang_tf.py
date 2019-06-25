import pickle
import numpy as np
import sys
from fen.classifier import Classifier
from fen.zhang_tf import ZhangDependencyModel

if __name__ == '__main__':
    import pickle
    import numpy as np
    import sys
    filePathEncodedTREC = sys.argv[1]
    filePathWord2vec = sys.argv[2]

    with open(filePathEncodedTREC, "rb") as f:
        (X_train, y_train), (X_eval, y_eval) = pickle.load(f)
    with open(filePathWord2vec, 'rb') as file:
        word_vector = np.load(file)

    word_vector = word_vector.astype('float32')
    sentence_length = X_train[0].shape[0]
    num_classes = y_train[0].shape[0]

    model = ZhangDependencyModel(
        embeddings=[word_vector])

    classifier = Classifier(
        model=model,
        input_length=sentence_length,
        output_length=num_classes)

    classifier.compile(batch_size=32)
    classifier.summary()
    classifier.train(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        epochs=20
    )

    print("Predictions:", classifier.predict(X_train[0:2]))
    print("Real:", y_train[0:2])
