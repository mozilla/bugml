import importlib
import configparser
import sys
import random
import math
import os
import io
import gc
import re
import json
import time
from time import strftime
import datetime as dt
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
from functools import reduce

from bs4 import BeautifulSoup

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.models import model_from_json

from keras import backend as K

import sklearn.metrics
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec

import configuration as config
from logger import Logger

from configuration import DATA_DIR_C
from configuration import MODELS_DIR_C
from configuration import LOGS_DIR_C

MAX_SEQUENCE_LENGTH = 4000
MAX_NB_WORDS = 5000
EMBEDDING_DIM = 100

VALIDATION_SPLIT = 0.3
MIN_PRODUCT_DESCRIPTIONS = 50
MAX_PRODUCT_DESCRIPTIONS = 500000
MIN_COMPONENT_DESCRIPTIONS = 50
MAX_COMPONENT_DESCRIPTIONS = 500000
CLASS_PRODUCT_NAME = config.PRODUCT
CLASS_COMPONENT_NAME = config.COMPONENT

BaseComponents = {
    'Firefox': ['Untriaged', 'Developer Tools: Debugger', 'Session Restore', 'Developer Tools'],
    'Core': ['DOM', 'General', 'XPCOM', 'Widget: Gtk'],
    'Firefox for iOS': ['General'],
    'NSS': ['Libraries']
}


def getKey(index, someDict):
    return [key for key, value in someDict.items() if value == index][0]


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


def balanse(datalist, nCount=1000):
    return datalist


def from_categorical(y_labels):
    return np.argmax(y_labels, 1)


def get_best_categories(y_labels, nCount=3):
    res = []
    for row in y_labels:
        ind = np.argpartition(row, -nCount)[-nCount:]
        inds = ind[np.argsort(row[ind])][::-1]
        res.append({key: row[key] for key in inds})
    return res


def dict_len(d1):
    nCount = 0
    for key, value in d1.items():
        nCount += len(value)
    return nCount


def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o: (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    same = set(o for o in intersect_keys if d1[o] == d2[o])
    return added, removed, modified, same


def dict_save(d1, dictName='dict_temp_save', sDir=DATA_DIR_C):
    np.save(os.path.join(sDir, dictName + '.npy'), d1)


def dict_load(dictName='dict_temp_save', sDir=DATA_DIR_C):
    return np.load(os.path.join(sDir, dictName + '.npy')).item()


def dataset_save(x_train, y_train, x_val, y_val, x_test, y_test,
                 datasetName='current_dataset', sDir=DATA_DIR_C):
    np.savez(os.path.join(sDir, datasetName + '.npz'),
             x_train=x_train,
             y_train=y_train,
             x_val=x_val,
             y_val=y_val,
             x_test=x_test,
             y_test=y_test)

# Warning: unlike dataset_save, this function is more generalized, based on makeDataset.
# Returns lists with different structure, based on splitCount.


def dataset_load(datasetName='current_dataset', sDir=DATA_DIR_C, splitCount=1):
    if splitCount > 0:
        if splitCount == 1:
            datasetFile = os.path.join(sDir, datasetName + '.npz')
            if os.path.exists(datasetFile):
                dataset = np.load(datasetFile)
                return list(map(lambda x: dataset[x], dataset.files))
        else:
            resv = []
            for i in range(splitCount):
                dataset = np.load(os.path.join(sDir, datasetName + '_' +
                                               str(splitCount) + '_' + str(i) + '.npz'))
                resv.append(list(map(lambda x: dataset[x], dataset.files)))
            return resv
    else:
        return [dict_load(dictName=datasetName + '_data', sDir=sDir),
                dict_load(dictName=datasetName + '_labels', sDir=sDir)]


def load_datasets(sDir=DATA_DIR_C):
    datasetsFiles = [f for f in os.listdir(sDir) if os.path.isfile(
        os.path.join(sDir, f)) and f.endswith('.npz')]
    datasets = {s[:-len('.npz')]: dataset_load(s[:-len('.npz')], sDir) for s in datasetsFiles}
    return datasets


def embedding_bin2txt(sName='GoogleNews-vectors-negative300.bin', sDir=DATA_DIR_C):
    model = KeyedVectors.load_word2vec_format(os.path.join(sDir, sName), binary=True)
    model.wv.save_word2vec_format('GoogleNews-vectors-negative300.txt')
    del model


def embedding_matrix_load(wordIndex, sName='glove.6B.' + str(EMBEDDING_DIM) + 'd.txt',
                          sDir=DATA_DIR_C, binary=False):
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


def save_models(models, sDir=MODELS_DIR_C):
    for name, model in models.items():
        if model != None:
            model_save(model, name, sDir)


def load_models(sDir=MODELS_DIR_C, sStartsWith=''):
    modelsFiles = []
    if sStartsWith == '':
        modelsFiles = [f for f in os.listdir(sDir) if os.path.isfile(
            os.path.join(sDir, f)) and f.endswith('.json')]
    else:
        modelsFiles = [f for f in os.listdir(sDir) if (os.path.isfile(os.path.join(sDir, f)) and f.endswith('.json')
                                                       and f.startswith(sStartsWith))]
    models = {s[:-len('.json')]: model_load(s[:-len('.json')], sDir) for s in modelsFiles}
    modelsScripts = []
    if sStartsWith == '':
        modelsScripts = [f for f in os.listdir(sDir) if os.path.isfile(
            os.path.join(sDir, f)) and f.endswith('.py')]
    else:
        modelsScripts = [f for f in os.listdir(sDir) if (os.path.isfile(os.path.join(sDir, f)) and f.endswith('.py')
                                                         and f.startswith(sStartsWith))]
    for s in modelsScripts:
        models[s[:-len('.py')]] = None
    return models


def prepareData(sDir=DATA_DIR_C, sName='bugData.csv', maxClasses=100, exactClasses={},
                minProductDescriptions=MIN_PRODUCT_DESCRIPTIONS,
                maxProductDescriptions=MAX_PRODUCT_DESCRIPTIONS,
                minComponentDescriptions=MIN_COMPONENT_DESCRIPTIONS,
                maxComponentDescriptions=MAX_COMPONENT_DESCRIPTIONS,
                otherPercents=50,
                balanseVirtual=True,
                balanseBinary=False):

    VIRTUAL_CLASS_NAME = 'Virtual Class'
    SELECTED_CLASS_NAME = 'Selected Class'

    if otherPercents < 0 or otherPercents >= 100:
        raise AssertionError('invalid otherPercents value: ' + str(otherPercents))

    product_data = {}

    data_frame = pd.read_csv(os.path.join(sDir, sName), low_memory=False)
    print(data_frame.shape)
    print(data_frame.columns)
    data_frame.dropna(subset=['description'], inplace=True)
    print(data_frame.shape)
    data_frame.drop([data_frame.columns[0]], axis=1, inplace=True)
    print(data_frame.shape)
    print(data_frame.columns)

    productData = {}
    productKeys = {}

    exactKeyNames = []
    if (isinstance(exactClasses, dict)):
        for key, vClasses in exactClasses.items():
            for item in vClasses:
                if isinstance(item, str):
                    exactKeyNames.append(key + ' - ' + item)
                else:
                    if isinstance(item, list):
                        if len(item) == 0:
                            exactKeyNames.append({key: item})
    else:
        if isinstance(exactClasses, list):
            for item in exactClasses:
                exactKeyNames.append(item)

    prod1 = data_frame[CLASS_PRODUCT_NAME].value_counts()
    print(prod1.shape)
    data_frame[data_frame[CLASS_PRODUCT_NAME].isin(prod1.where(prod1 < minProductDescriptions)
                                                   .dropna().index.tolist())] = 'Other ' + CLASS_PRODUCT_NAME + 's'
    prod1 = data_frame[CLASS_PRODUCT_NAME].value_counts()
    print(prod1)
    prodNames = prod1.index.tolist()
    for item in prod1.where(prod1 > maxProductDescriptions).dropna().index.tolist():
        prodNames.remove(item)
    n1 = len(prodNames)

    productData = {name: data_frame[data_frame[CLASS_PRODUCT_NAME] == name] for name in prodNames}

    keyIndex = 0
    product_data[VIRTUAL_CLASS_NAME] = []
    for name, prod_frame in productData.items():
        comp1 = prod_frame[CLASS_COMPONENT_NAME].value_counts()
        prod_frame[prod_frame[CLASS_COMPONENT_NAME].isin(comp1.where(comp1 < minComponentDescriptions)
                                                         .dropna().index.tolist())] = 'Other ' + CLASS_COMPONENT_NAME + 's'
        comp1 = prod_frame[CLASS_COMPONENT_NAME].value_counts()
        compNames = comp1.index.tolist()
        for item in comp1.where(comp1 > maxComponentDescriptions).dropna().index.tolist():
            compNames.remove(item)
            print('removed: ' + str(item))
        n2 = len(compNames)

        for key in compNames:
            items = []
            if 'opendate' in prod_frame.columns:
                items = [{'id': x0, 'date': x6, 'data': ' ||| '.join([x1, x2, x3, x4, x5])}
                         for x0, x1, x2, x3, x4, x5, x6 in zip(
                    prod_frame[prod_frame[CLASS_COMPONENT_NAME] == key]['bug_id'].values.tolist(),
                    prod_frame[prod_frame[CLASS_COMPONENT_NAME]
                               == key]['reporter'].values.tolist(),
                    prod_frame[prod_frame[CLASS_COMPONENT_NAME]
                               == key]['short_desc'].values.tolist(),
                    prod_frame[prod_frame[CLASS_COMPONENT_NAME] == key]['op_sys'].values.tolist(),
                    prod_frame[prod_frame[CLASS_COMPONENT_NAME]
                               == key]['rep_platform'].values.tolist(),
                    list(map(lambda s: str(s),
                             prod_frame[prod_frame[CLASS_COMPONENT_NAME] == key]['description'].values.tolist())),
                    prod_frame[prod_frame[CLASS_COMPONENT_NAME] == key]['opendate'].values.tolist())]
            else:
                items = [{'id': x0, 'date': None, 'data': ' ||| '.join([x1, x2, x3, x4, x5])}
                         for x0, x1, x2, x3, x4, x5 in zip(
                    prod_frame[prod_frame[CLASS_COMPONENT_NAME] == key]['bug_id'].values.tolist(),
                    prod_frame[prod_frame[CLASS_COMPONENT_NAME]
                               == key]['reporter'].values.tolist(),
                    prod_frame[prod_frame[CLASS_COMPONENT_NAME]
                               == key]['short_desc'].values.tolist(),
                    prod_frame[prod_frame[CLASS_COMPONENT_NAME] == key]['op_sys'].values.tolist(),
                    prod_frame[prod_frame[CLASS_COMPONENT_NAME]
                               == key]['rep_platform'].values.tolist(),
                    list(map(lambda s: str(s),
                             prod_frame[prod_frame[CLASS_COMPONENT_NAME] == key]['description'].values.tolist())))]
            if isinstance(exactClasses, dict):
                if len(exactKeyNames) > 0:
                    if (name + ' - ' + key in exactKeyNames) or ({name: []} in exactKeyNames):
                        product_data[name + ' - ' + key] = items
                    else:
                        if len(product_data[VIRTUAL_CLASS_NAME]) > 0:
                            product_data[VIRTUAL_CLASS_NAME].extend(items)
                        else:
                            product_data[VIRTUAL_CLASS_NAME] = items
                else:
                    if key != 'Other ' + CLASS_COMPONENT_NAME + 's':
                        product_data[name + ' - ' + key] = items
                    else:
                        if len(product_data[VIRTUAL_CLASS_NAME]) > 0:
                            product_data[VIRTUAL_CLASS_NAME].extend(items)
                        else:
                            product_data[VIRTUAL_CLASS_NAME] = items
                productKeys[name + ' - ' + key] = keyIndex
                keyIndex += 1
            else:
                if isinstance(exactClasses, list):
                    if len(exactKeyNames) > 0:
                        if name in exactKeyNames:
                            if name in product_data:
                                product_data[name].extend(items)
                            else:
                                product_data[name] = items
                        else:
                            if len(product_data[VIRTUAL_CLASS_NAME]) > 0:
                                product_data[VIRTUAL_CLASS_NAME].extend(items)
                            else:
                                product_data[VIRTUAL_CLASS_NAME] = items
                    else:
                        if key != 'Other ' + CLASS_COMPONENT_NAME + 's':
                            if name in product_data:
                                product_data[name].extend(items)
                            else:
                                product_data[name] = items
        if isinstance(exactClasses, list):
            productKeys[name] = keyIndex
            keyIndex += 1
    if keyIndex <= maxClasses:
        if balanseVirtual:
            baseDataCount = 0
            otherDataCount = 0
            for key, texts in product_data.items():
                if key == VIRTUAL_CLASS_NAME:
                    otherDataCount += len(texts)
                else:
                    baseDataCount += len(texts)
            print('baseDataCount = ', baseDataCount)
            print('otherDataCount = ', otherDataCount)
            # virtualCount / (virtualCount + baseDataCount) = otherPercents / 100
            virtualCount = int(((baseDataCount * otherPercents) / 100.0) /
                               (1.0 - otherPercents / 100.0))
            print('virtualCount = ', virtualCount)
            if otherDataCount > virtualCount:
                print('balansing...')
                product_data[VIRTUAL_CLASS_NAME] = random.sample(
                    product_data[VIRTUAL_CLASS_NAME], virtualCount)

    else:
        other_product_data = dict(
            sorted(product_data.items(), key=lambda k: len(k[1]), reverse=True)[maxClasses+1:])
        for value in other_product_data.values():
            product_data[VIRTUAL_CLASS_NAME].extend(value)
        product_data = dict(sorted(product_data.items(),
                                   key=lambda k: len(k[1]), reverse=True)[:maxClasses])
        if balanseVirtual:
            otherDataCount = len(product_data[VIRTUAL_CLASS_NAME])
            baseDataCount = dict_len(product_data) - otherDataCount
            print('baseDataCount = ', baseDataCount)
            print('otherDataCount = ', otherDataCount)
            # virtualCount / (virtualCount + baseDataCount) = otherPercents / 100
            virtualCount = int(((baseDataCount * otherPercents) / 100.0) /
                               (1.0 - otherPercents / 100.0))
            print('virtualCount = ', virtualCount)
            if otherDataCount > virtualCount:
                print('balansing...')
                product_data[VIRTUAL_CLASS_NAME] = random.sample(
                    product_data[VIRTUAL_CLASS_NAME], virtualCount)

    if balanseVirtual and (otherPercents == 0):
        product_data.pop(VIRTUAL_CLASS_NAME)
    productKeys = {list(product_data.keys())[i]: i for i in range(len(product_data))}

    if balanseBinary:
        selectedItems = []
        for key, texts in product_data.items():
            if key != VIRTUAL_CLASS_NAME:
                selectedItems.extend(texts)
        product_data = {
            VIRTUAL_CLASS_NAME: product_data[VIRTUAL_CLASS_NAME], SELECTED_CLASS_NAME: selectedItems}
        productKeys = {SELECTED_CLASS_NAME: 0, VIRTUAL_CLASS_NAME: 1}

    data_frame = pd.DataFrame()
    gc.collect()
    print('prepareData finished')
    return product_data, productKeys


def makeDataset(data_dict, productKeys, datasetName='current_dataset', sDir=DATA_DIR_C,
                splitCount=1, splitVirtualOnly=False, maxCount=0):
    print('make dataset: ', datasetName)
    texts = []
    labels = []
    print('productKeys len = ', len(productKeys))
    print('data_dict len = ', len(data_dict))

    if splitVirtualOnly == False:
        for key, value in data_dict.items():
            for sDesc in value:
                text = BeautifulSoup(sDesc['data'], "lxml")
                sText = clean_str2(text.get_text())
                if sText not in texts:
                    texts.append(sText)
                    labels.append(productKeys[key])
            print('converting text for key: ' + str(key) +
                  ' with len = ' + str(len(value)) + ' finished')

        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

        labels = to_categorical(np.asarray(labels))
        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)

        indices = np.random.choice(data.shape[0],
                                   maxCount if maxCount > 0 and data.shape[0] > maxCount else data.shape[0],
                                   replace=False)
        # np.random.shuffle(indices)

        data = data[indices]
        labels = labels[indices]

        print('Actual shape of data tensor:', data.shape)
        print('Actual shape of label tensor:', labels.shape)

        nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

        if splitCount > 0:
            x_train = data[:-nb_validation_samples]
            y_train = labels[:-nb_validation_samples]
            x_val = data[-nb_validation_samples: -(nb_validation_samples // 2)]
            y_val = labels[-nb_validation_samples: -(nb_validation_samples // 2)]
            x_test = data[-(nb_validation_samples // 2):]
            y_test = labels[-(nb_validation_samples // 2):]
            dataset_save(x_train, y_train, x_val, y_val, x_test, y_test,
                         datasetName=datasetName, sDir=sDir)
            if splitCount > 1:
                def makePart(X, n, i):
                    return X[(i*len(X))//n: ((i+1)*len(X))//n]
                for i in range(splitCount):
                    dataset_save(makePart(x_train, splitCount, i),
                                 makePart(y_train, splitCount, i),
                                 makePart(x_val, splitCount, i),
                                 makePart(y_val, splitCount, i),
                                 makePart(x_test, splitCount, i),
                                 makePart(y_test, splitCount, i),
                                 datasetName=datasetName + '_' + str(splitCount) + '_' + str(i),
                                 sDir=sDir)
        else:
            dict_save(data, datasetName + '_data', sDir=sDir)
            dict_save(labels, datasetName + '_labels', sDir=sDir)

        dict_save(word_index, datasetName + '_word_index', sDir=sDir)
    else:
        raise NotImplementedError('''May be used in future to create ensembles with model, 
                                     trained on few datasets with equal base classes,
                                     but different parts of virtual classes data''')

    print('makeDataset finished')
#    print([np.array_equal(x, y) for x,y in zip([x_train, y_train, x_val, y_val, x_test, y_test], dataset_load())])


def testModel(resultsFrame, modelName='current_model', datasetName='current_dataset',
              dataDir=DATA_DIR_C,
              modelsDir=MODELS_DIR_C):
    bModelReady = (os.path.exists(os.path.join(modelsDir, modelName + '.json')) and
                   os.path.exists(os.path.join(modelsDir, modelName + '.h5')))
    if bModelReady:
        model = model_load(modelName=modelName, sDir=modelsDir)
        x_train, y_train, x_val, y_val, x_test, y_test = dataset_load(
            datasetName=datasetName, sDir=dataDir)
        try:
            scores = model.evaluate(x_test, y_test)
            resultsFrame.set_value(modelName, datasetName, scores[1])
        except Exception as exc:
            print(exc)
    else:
        try:
            mfile = importlib.import_module(name=modelName)
            importlib.reload(mfile)
            try:
                mfile.set_params(params)
            except Exception as exc2:
                print(exc2)
            resultsFrame.set_value(modelName, datasetName, mfile.test_model(
                modelName, datasetName, x_test, y_test))
        except Exception as exc:
            print(exc)


def reloadSession():
    K.clear_session()
    gc.collect()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))


#    Params:
# testData - array of dicts like
# {
#  'id' : bug_id,
#  'date': '2000-01-01 15:00:00',
#  'data': 'reporter ||| short_desc ||| op_sys ||| platform ||| description'
# }
# pKeys - dict with product keys (classes), like:
# {
#  'Firefox - General': 0,
#  'Firefox - Installer': 1,
#  ...
# }
# pData (optional) - product data from global dataset with dict, which has product name key and product data in same format as testData:
# {'Firefox - Untriaged' : testData}
#
# modelName - name of model, which will be loaded.
# Func will search python file with the same name (+ '.py') and will call function
#  predict_components(testData, modelName, pKeys, pData)
# with the same parameters,
# which can return matrix with probabilities for classes for testData, similar to returned by keras model predict function.
# or can return list with dicts with next structure:
# [{
#  'bug_id': 0,
#  'Firefox - Installer': 0.7,
#  'Firefox - General': 0.15,
#  'Firefox - Sync': 0.1,
# }] # dict must contain bug_id with int value and string components with float or int probabilities.
# If python file with correct model cannot be found, keras model will be used instead.
#
#     Returns:
# DataFrame with bugs id, data, date and classes columns.
def predict_classes(modelName, testData, pKeys, pData=None):
    res = pd.DataFrame()
    res['data'] = [item['data'] for item in testData]
    res['id'] = [item['id'] for item in testData]
    res['date'] = [item['date'] for item in testData]
    res['classes'] = None
    res.set_index(['id'], inplace=True, drop=False)

    cat_labels = None

    try:
        mfile = importlib.import_module(name=modelName)
        importlib.reload(mfile)
        cat_labels = mfile.predict_components(testData, modelName, pKeys, pData)
        if isinstance(cat_labels, list):
            bValid = True
            for item in cat_labels:
                if not (isinstance(item, dict) or len(dict) > 1):
                    bValid = False
                    break
                if not ('bug_id' in item):
                    bValid = False
                    break
                if not isinstance(item['bug_id'], int):
                    bValid = False
                    break
                for key, value in item.items():
                    if not (isinstance(key, str) and isinstance(value, (float, int))):
                        bValid = False
                        break
                if not bValid:
                    break
            if bValid:
                for item in cat_labels:
                    res.set_value(item['bug_id'], 'classes',
                                  ', '.join([category + '(' + ('%.6f' % percent) + ')'
                                             for category, percent in item.items() if category != 'bug_id']))
                return res
    except Exception as exc2:
        print('predict_classes exception: ')
        print(exc2)

    if not isinstance(cat_labels, (np.ndarray, np.generic)):
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

    if not isinstance(cat_labels, (np.ndarray, np.generic)):
        raise AssertionError('Error during predict categories of bugs')

    labels = get_best_categories(cat_labels, 3)
    res['classes'] = [', '.join([getKey(index, pKeys) + '(' + ('%.6f' % percent) + ')'
                                 for index, percent in item.items()])
                      for item in labels]
    return res
