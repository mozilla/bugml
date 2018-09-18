import os
import gc
import datetime as dt
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Average, SpatialDropout1D, LeakyReLU
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, GlobalMaxPool1D
from keras import backend as K
import keras.callbacks
import keras.optimizers

from sklearn.metrics import roc_auc_score

DATA_DIR_C = os.path.join(os.getcwd(), 'data')
MODELS_DIR_C = os.path.join(os.getcwd(), 'models')
LOGS_DIR_C = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(DATA_DIR_C):
    os.makedirs(DATA_DIR_C)
if not os.path.exists(MODELS_DIR_C):
    os.makedirs(MODELS_DIR_C)
if not os.path.exists(LOGS_DIR_C):
    os.makedirs(LOGS_DIR_C)

ModelParams = {'batch_size': 64,
               'dropout': 0.1,
               'conv_size': 3,
               'conv_filters': 256,
               'epochs': 12,
               'optimizer': 'nadam',
               'pooling': 'local',
               'padding': 'same',
               'activation': 'relu',
               'use_pretrained': 1,
               'name_suffix': 'cnn1'
               }


def getModelParamData(sName):
    s = ModelParams[sName]
    kStart = s.find('(')
    kEnd = s.rfind(')')
    res = []
    if kStart != -1 and kEnd != -1 and (kEnd > kStart + 1):
        res.append(s[:kStart])
        res += s[kStart + 1: kEnd].split(',')
    else:
        res.append(s)
    return res


MAX_SEQUENCE_LENGTH = 4000
MAX_NB_WORDS = 5000
EMBEDDING_DIM = 100


def set_params(params):
    try:
        ModelParams.update(params)
    except Exception as exc:
        pass


def clean_str2(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def dict_save(d1, dictName='dict_temp_save', sDir=DATA_DIR_C):
    np.save(os.path.join(sDir, dictName + '.npy'), d1)


def dict_load(dictName='dict_temp_save', sDir=DATA_DIR_C):
    return np.load(os.path.join(sDir, dictName + '.npy')).item()


def dataset_load(datasetName='current_dataset', sDir=DATA_DIR_C):
    dataset = np.load(os.path.join(sDir, datasetName + '.npz'))
    return list(map(lambda x: dataset[x], dataset.files))


def embedding_matrix_load(wordIndex, sName='glove.6B.100d.txt', sDir=DATA_DIR_C, binary=False):
    embeddings_index = {}
    word2vec = None
    if binary:
        word2vec = KeyedVectors.load_word2vec_format(os.path.join(sDir, sName), binary=True)
    else:
        with open(os.path.join(sDir, sName), mode='r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    embedding_matrix = np.random.random((len(wordIndex) + 1, EMBEDDING_DIM))
    if binary:
        for word, i in wordIndex.items():
            try:
                embedding_matrix[i, :] = word2vec[word]
            except KeyError:
                pass
    else:
        for word, i in wordIndex.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    gc.collect()
    return embedding_matrix


def model_save(model, modelName='model_temp', sDir=MODELS_DIR_C):
    if modelName == '':
        modelName = model.name
    model.name = modelName
    model_json = model.to_json()
    with open(os.path.join(sDir, modelName + ".json"), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(sDir, modelName + ".h5"))


def model_load(modelName='model_temp', sDir=MODELS_DIR_C, bCompile=True):
    with open(os.path.join(sDir, modelName + '.json'), 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(sDir, modelName + ".h5"))
    if modelName != '':
        loaded_model.name = modelName
    if bCompile:
        loaded_model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['acc'])
    return loaded_model


def reloadSession():
    K.clear_session()
    gc.collect()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))


def test_model(modelName, datasetName, x_test, y_test, sequenceInput=None):
    x_train, y_train, x_val, y_val, x_test, y_test = dataset_load(datasetName)
    word_index = dict_load(datasetName + '_word_index')

    embedding_layer = None
    if ModelParams['use_pretrained'] == 1:
        embedding_matrix = embedding_matrix_load(word_index)
        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=True)
    else:
        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=True)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_sdrop1 = SpatialDropout1D(ModelParams['dropout'])(embedded_sequences)

    l_conv1 = None
    l_sdrop2 = None
    l_act1 = None
    l_pool1 = None
    preds = None

    if ModelParams['activation'] == 'relu':
        l_conv1 = Conv1D(ModelParams['conv_filters'], ModelParams['conv_size'],
                         padding=ModelParams['padding'], activation='relu')(l_sdrop1)
        l_sdrop2 = SpatialDropout1D(ModelParams['dropout'])(l_conv1)
    else:
        acts = getModelParamData('activation')
        l_conv1 = Conv1D(ModelParams['conv_filters'], ModelParams['conv_size'],
                         padding=ModelParams['padding'])(l_sdrop1)
        if len(acts) == 1:
            if acts[0].lower() == 'leakyrelu':
                l_act1 = LeakyReLU(0.3)(l_conv1)
        if len(acts) > 1:
            if acts[0].lower() == 'leakyrelu':
                l_act1 = LeakyReLU(acts[1])(l_conv1)
        if l_act1 == None:
            print('Cannot parse activation parameter. LeakyReLU layer with default param(0.3) will be used in model')
            l_act1 = LeakyReLU(0.3)(l_conv1)
        l_sdrop2 = SpatialDropout1D(ModelParams['dropout'])(l_act1)

    if ModelParams['pooling'] == 'global':
        l_pool1 = GlobalMaxPool1D()(l_sdrop2)
        preds = Dense(y_val.shape[1], activation='softmax')(l_pool1)
    else:
        if ModelParams['padding'] == 'same':
            l_pool1 = MaxPooling1D(MAX_SEQUENCE_LENGTH)(l_sdrop2)
        else:
            l_pool1 = MaxPooling1D(MAX_SEQUENCE_LENGTH - ModelParams['conv_size'] + 1)(l_sdrop2)
        l_flat = Flatten()(l_pool1)
        preds = Dense(y_val.shape[1], activation='softmax')(l_flat)

    print('training model ' + modelName + ' on dataset ' + datasetName + ' start')
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer=ModelParams['optimizer'],
                  metrics=['acc'])

    model.summary()
    try:
        model.fit(x_train, y_train, validation_data=(x_val, y_val),
                  epochs=ModelParams['epochs'], batch_size=ModelParams['batch_size'],
                  callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=1, verbose=1,
                                                           mode='auto')])
    except Exception as exc:
        print(exc)
        model_save(model, modelName + '_temp')
    print('trainModels finished')

    sfName = modelName + '_' + datasetName
    for key, value in ModelParams.items():
        sfName += ('_' + str(value))
    model_save(model, sfName)
    scores = model.evaluate(x_test, y_test)
    print(scores)
    print('training model finished')
    del model
    reloadSession()
    return scores[1]


def predict_components(testData, modelName, pKeys, pData):
    bModelReady = (os.path.exists(os.path.join(MODELS_DIR_C, modelName + '.json')) and
                   os.path.exists(os.path.join(MODELS_DIR_C, modelName + '.h5')))
    if bModelReady:
        texts = []
        for value in testData:
            text = BeautifulSoup(value['data'], "lxml")
            sText = clean_str2(text.get_text())
            texts.append(sText)

        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

        model = model_load(modelName=modelName, bCompile=True)
        cat_labels = model.predict(data)
        del model
        reloadSession()
        return cat_labels
    return None
