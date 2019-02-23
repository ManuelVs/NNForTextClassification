# Repositorio de pruebas en ML y DL

## /data

- data.pickle: Datos del TREC guardados en el formato pickle.
  (Xtrain, ytrain), (Xtest, ytest)
- word2vec.npz: Un embedding creado con word2vec. Es una matriz numpy.
- Xavier.npz: Un embedding inicializado aleatoriamente con la
  distribución xavier. Es una matriz numpy.

## /fen

Modelos implementados usando exclusivamente tensorflow.
Existe una clase 'Classifier', que engloba un modelo, es capaz de
entrenarlo y sacar estadísticas.

Actualmente solo hay 3 modelos implementados
- Kim CNN: Basado en el modelo <https://arxiv.org/pdf/1408.5882.pdf>.
- Shin CCNN: Basado en el modelo <https://ieeexplore.ieee.org/document/8367159>.
- LSTM: Un modelo utilizando celdas LSTM, simplemente de pruebas.

## /tests

Ejemplos de uso de esta biblioteca.