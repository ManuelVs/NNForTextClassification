# Repositorio de pruebas en ML y DL para clasificación de texto

## /data

- data.pickle: Datos del TREC-6 guardados en el formato pickle.
  (Xtrain, ytrain), (Xtest, ytest)
- word2vec.npz: Un embedding creado con word2vec. Es una matriz numpy.
- Xavier.npz: Un embedding inicializado aleatoriamente con la
  distribución xavier. Es una matriz numpy.

## /fen

Modelos implementados usando exclusivamente tensorflow.
Existe una clase 'Classifier', que engloba un modelo, es capaz de
entrenarlo y sacar estadísticas.

Actualmente hay 5 modelos implementados
- Yoon Kim, CNN: Basado en el modelo <https://arxiv.org/pdf/1408.5882.pdf>.
- Joongbo Shin, C-CNN: Basado en el modelo <http://milab.snu.ac.kr/pub/BigComp2018.pdf>.
- Chunting Zhou, C-LSTM: Basado en el modelo <https://arxiv.org/pdf/1511.08630>.
- Rui Zhang, LSTM-CNN: Basado en el modelo <https://arxiv.org/pdf/1611.02361.pdf>.
- Peng Zhou, BLSTM-CNN: Basado en el modelo <https://arxiv.org/pdf/1611.06639v1.pdf>.
- LSTM: Un modelo simple utilizando celdas LSTM, simplemente de pruebas.

## /tests

Ejemplos de uso de esta biblioteca, además de pequeños tests.
