import importlib
import configparser
import sys
import threading
import random
import math
import base64
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
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import reduce

import configuration as config
from logger import Logger
from storage import Storage

from configuration import DATA_DIR_C
from configuration import MODELS_DIR_C
from configuration import LOGS_DIR_C
from configuration import CONFIG_DIR_C
from configuration import CONFIG_FILENAME_C
from configuration import UNTRIAGED_FILE_C

from configuration import cfgModelsSection
from configuration import cfgDownloadSection

from configuration import cfgElementProducts
from configuration import cfgElementInterval
from configuration import cfgElementState

from configuration import cfgValueStateInactive
from configuration import cfgValueStateBusy
from configuration import cfgValueStateStop

from DataDownloader import *
from DataProcessor import * 


ParametersRanges = {'batch_size': [32],
                    'dropout': [0.15],
                    'conv_size': [3],
                    'epochs': [4],
                    'optimizer': ['nadam'],
                    'conv_filters': [256],               
                    'pooling': ['local'],
                    'padding': ['same'],
                    'activation': ['relu'], #, 'LeakyRelu(0.2)'
                    'lstm_units': [80],
                    'use_pretrained': [0], # use pretrained glove vector (1) or train from zero (0).
                    'split_count': [1], # 1 for single model, > 1 for ensemble of models, trained on parts of dataset.
                    'name_suffix': ['suffixCnn2'] # just a name suffix. May be used to train same models few times.
                   }

'''
Config file example:

[Download]
UntriagedProducts=Firefox
Interval=3600
State=Inactive
'''

def asListJson(value):
    if isinstance(value, str):
        if value[0] == '[':
            return json.loads(value)
        else:
            value = filter(None, [x.strip() for x in value.splitlines()])
    return list(value)

def asList(value, flatten=True):
    values = asListJson(value)
    if not flatten:
        return values
    result = []
    if isinstance (values, list):
        for value in values:
            subvalues = value.split()
            result.extend(subvalues)
    return result

# add default config values if they aren't present in config
def checkConfig(Config):
    if cfgDownloadSection not in Config.sections():
        Config.add_section(cfgDownloadSection)
        Config.set(cfgDownloadSection, cfgElementProducts, 'Firefox')
        Config.set(cfgDownloadSection, cfgElementInterval, '3600')
        Config.set(cfgDownloadSection, cfgElementState, cfgValueStateInactive) # States: 'Inactive', 'Busy', 'Stop'
    if cfgModelsSection not in Config.sections():
        Config.add_section(cfgModelsSection)

# config file will be created with default values if not exists or incorrect.
def readFromConfig(entrySection, entryName, logger = None,
                   configName = CONFIG_FILENAME_C, configDir = CONFIG_DIR_C):
    res = []
    try:
        sConfigFile = os.path.join(configDir, configName)
        if os.path.exists(sConfigFile):
            Config = configparser.ConfigParser()
            Config.read(sConfigFile)
            res = asList(Config.get(entrySection, entryName))        
        if len(res) == 0:
            Config = configparser.ConfigParser()
            checkConfig(Config)
            with open(sConfigFile, 'w') as f:
                Config.write(f)
            res = asList(Config.get(entrySection, entryName))
    except Exception as exc:
        if logger is None:
            print('Error in readFromConfig: ' + str(exc))
        else:
            logger.error('Error in readFromConfig: ' + str(exc))
    return res

def writeToConfig(entrySection, entryName, entryValue, logger = None,
                  configName = CONFIG_FILENAME_C, configDir = CONFIG_DIR_C):
    try:
        sConfigFile = os.path.join(configDir, configName)
        if os.path.exists(sConfigFile):
            Config = configparser.ConfigParser()
            Config.read(sConfigFile)
            checkConfig(Config)
            if entrySection not in Config.sections():
                Config.add_section(entrySection)
            if len(entryName) > 0:
                Config.set(entrySection, entryName, entryValue)
            with open(sConfigFile, 'w') as f:
                Config.write(f)
        else:
            Config = configparser.ConfigParser()
            checkConfig(Config)
            if entrySection not in [cfgDownloadSection, cfgModelsSection]:
                Config.add_section(entrySection)
            if len(entryName) > 0:
                Config.set(entrySection, entryName, entryValue)
                if logger is None:
                    print('Changed config value: Section: ' + entrySection + 
                          ' Name: ' + entryName + ' Value: ' + entryValue)
                else:
                    logger.info('Changed config value: Section: ' + entrySection + 
                                ' Name: ' + entryName + ' Value: ' + entryValue)
            with open(sConfigFile, 'w') as f:
                Config.write(f)
    except Exception as exc:
        if logger is None:
            print('Error in writeToConfig: ' + str(exc))
        else:
            logger.error('Error in writeToConfig: ' + str(exc))

            
def getParams(sName, index = 0, ModelParams = ParametersRanges):
    return ModelParams[sName][index].split('_')

def makeModelName(params, modelName = 'model_opt_py', datasetName = 'current_dataset'):
    sfName = modelName + '_' + datasetName
    for key, value in params.items():
        sfName += ('_' + str(value))
    return sfName

def checkSameListElements(list1, list2):
    if isinstance(list1, list) and isinstance(list2, list):
        try:
            return set(list1) == set(list2)
        except Exception as exc:
            return sorted(list1) == sorted(list2)
    return False
    
def checkSameModels(model1, model2):
    if isinstance(model1, str) and isinstance(model2, str):
        try:
            if model1[-len('.json'):] == '.json':
                s1 = model1[:-len('.json')]
            else:
                s1 = model1
            if model2[-len('.json'):] == '.json':
                s2 = model2[:-len('.json')]
            else:
                s2 = model2
            return checkSameListElements(s1.split('_'), s2.split('_'))
        except Exception as exc:
            return False
    return False

def findSameModels(modelName, sDir = MODELS_DIR_C):
    return [f[:-len('.json')] for f in os.listdir(sDir) if (os.path.isfile(os.path.join(sDir, f))
                                                       and f.endswith('.json')
                                                       and checkSameModels(modelName, f))]

def writeModelsData(modelsData, fData = 'modelsInfo.txt', sDir = LOGS_DIR_C, bHistory = True):
    if bHistory:
        with open(os.path.join(sDir, fData), 'at') as f:
            for modelData in modelsData:
                json.dump(modelData, f, indent=4)
                print(file=f)
    else:
        with open(os.path.join(sDir, fData), 'wt') as f:
            json.dump(modelsData, f)
            
def loadModelsData(fData = 'modelsData.json', sDir = DATA_DIR_C):
    with open(os.path.join(sDir, fData), 'rt') as f:
        return json.load(f)


# In this function, you can select name of python file with model, which must contains a few functions
# def set_params(params):
# def test_model(modelName, datasetName, px_test, py_test, sequenceInput = None): # must return float score between 0 and 1.
# function will find best parameters for model to get best score.
def adaptiveModelSelector(logger,
                          modelNames = ['model_opt_py'],
                          datasetNames = ['FirefoxBugData'],
                          modelsDir = MODELS_DIR_C,
                          dataDir = DATA_DIR_C,
                          modelsParams = ParametersRanges,
                          configName = CONFIG_FILENAME_C,
                          configDir = CONFIG_DIR_C):
    
    sConfigFile = os.path.join(configDir, configName)
    Config = configparser.ConfigParser()
    if os.path.exists(sConfigFile):
        Config.read(sConfigFile)
        if cfgModelsSection not in Config.sections():
            Config.add_section(cfgModelsSection)
    else:
        checkConfig(Config)

        
    ldata = [len(v) for name, v in modelsParams.items()]
    nMax = reduce(lambda x, y: x * y, ldata)
    ldata2 = [reduce(lambda x, y: x * y, ldata[len(ldata):n1:-1], 1) for n1 in range(0, len(ldata) - 1)]
    
    def getNextParams(index):
        if index >= nMax or index < 0:
            logger.warning('getNextParams: index is out of range ' + str(index))
            return {}
        res = {}
        itemIndex = 0
        num = index
        print('num = ', num)
        for item, v in modelsParams.items():
            if itemIndex < len(ldata2):
                k = num // ldata2[itemIndex]
                print(itemIndex, k)
                res[item] = v[k]
                num -= k * ldata2[itemIndex]
            else:
                res[item] = v[num]
            itemIndex += 1
        print('res = ',res)
        return res

    if modelsDir not in sys.path:
        sys.path.append(modelsDir)
    
    for datasetName in datasetNames:
        _,_,_,_, x_test, y_test = dataset_load(datasetName)
        bestScore = 0
        bestParams = {}
        bestModel = ''
        bestModelFileName = ''
        for modelName in modelNames:
            logger.info('adaptiveModelSelector: test model: ' + modelName)
            mfile = importlib.import_module(modelName)
            index = 0
            while index < nMax:
                currentParams = getNextParams(index)
                if len(currentParams) > 0:
                    logger.info('Current parameters: ' + str(currentParams))
                    prevModels = [f[:-len('.json')] for f in os.listdir(modelsDir)
                                  if (os.path.isfile(os.path.join(modelsDir, f)) and f.endswith('.json'))]
                    prevModels.extend([f[:-len('.pkl')] for f in os.listdir(modelsDir) 
                                       if (os.path.isfile(os.path.join(modelsDir, f)) and f.endswith('.pkl'))])
                    
                    importlib.reload(mfile)
                    try:
                        mfile.set_params(currentParams)
                    except Exception as exc:
                        logger.info('Unsupported setting parameters. Finish after first iteration.' + str(exc))
                        if index > 0:
                            break
                    try:
                        curScore = mfile.test_model(modelName, datasetName, x_test, y_test)
                        logger.info('Current model: ' + modelName + '. Score: ' + str(curScore))
                    except Exception as exc:
                        logger.info('Unsupported test_model. Skip model ' + modelName + str(exc))
                        gc.collect()
                        break
                    if curScore > bestScore:
                        newModels = [f[:-len('.json')] for f in os.listdir(modelsDir)
                                     if (os.path.isfile(os.path.join(modelsDir, f)) and f.endswith('.json'))]
                        newModels.extend([f[:-len('.pkl')] for f in os.listdir(modelsDir) 
                                          if (os.path.isfile(os.path.join(modelsDir, f)) and f.endswith('.pkl'))])
                        
                        bestScore = curScore
                        bestParams = currentParams
                        bestModel = modelName
                        for item in set(newModels) - set(prevModels):
                            bestModelFileName = item
                        logger.info('Current best model score: ' + str(bestScore) + ' with model: ' + bestModel)
                        logger.info('Current bestModelFileName: ' + bestModelFileName)

                        
                    gc.collect()
                else:
                    logger.info('Current parameters: empty')
                index += 1
                if len(currentParams) <= 0:
                    break

            #print(bestScore)
            #print(bestParams)
            #print(bestModel)
            logger.info('best model score: ' + str(bestScore) + ' with model: ' + bestModel)
            logger.info('best model params: ' + str(bestParams))
            logger.info('bestModelFileName: ' + bestModelFileName)
            
        if len(bestModel) > 0:
            if os.path.exists(os.path.join(modelsDir, bestModelFileName + '.pkl')) or len(bestModelFileName) == 0:
                Config.set(cfgModelsSection, datasetName, bestModel)
            else:
                Config.set(cfgModelsSection, datasetName, bestModelFileName)
 
    with open(sConfigFile, 'w') as f:
        Config.write(f)


#  runBugsDataUpdater algorithm:
# - every period of time (1 hour by default) do next steps:
#   - load products from configuration
#   - download global bugs data (if not downloaded)
#   - download untriaged bugs data
#   - make datasets (from global) if not exists for selected products:
#   - select and compare models for new datasets
#   - save best models information
#   - make dataset from downloaded untriaged bugs,
#   - classify them with best model
#   - save it to database
#   - select untriaged bugs from db and check their components changes in bugzilla, update information in db.
def runBugsDataUpdater(updateIntervalDefault = 3600,
                       modelNames = ['model_bugclassifier', 'model_opt_py'],
                       modelsDir = MODELS_DIR_C,
                       dataDir = DATA_DIR_C,
                       modelsParams = ParametersRanges):
    updateInterval = updateIntervalDefault
    logger = Logger(log_dir=config.LOG_DIR, log_file_base_name='BugsDataUpdater')
    try:
        updateInterval = int(readFromConfig(cfgDownloadSection, cfgElementInterval)[0])
    except Exception:
        logger.warning('Incorrect update interval in config. Using default update interval = ' + str(updateIntervalDefault))
    busyState = cfgValueStateInactive
    try:
        busyState = readFromConfig(cfgDownloadSection, cfgElementState, logger = logger)
    except Exception:
        logger.warning('Incorrect update interval in config. Using default update interval = ' + str(updateIntervalDefault))
    
    def BugsDataUpdaterFunc():
        nonlocal busyState
        nonlocal logger
        busyState = readFromConfig(cfgDownloadSection, cfgElementState, logger = logger)
        if busyState == cfgValueStateBusy:
            return
        try:
            writeToConfig(cfgDownloadSection, cfgElementState, cfgValueStateBusy, logger = logger)
            if not os.path.exists(os.path.join(dataDir, 'bugDataTest500.csv')):
                downloadData(sName='bugDataTest500.csv', resolutions = ['FIXED', 'WONTFIX'], nMax=500000, baseLogger = logger)

            untriagedProducts = readFromConfig(cfgDownloadSection, cfgElementProducts)
            
            dbStorage = Storage(logger)
            if dbStorage.tryConnect():
                logger.info('Connected to storage')
                dbStorage.configureDB()
            else:
                logger.error('Cannot connect to storage')
            
            product_prev = {}
            keys_prev = {}
            if os.path.exists(os.path.join(DATA_DIR_C, UNTRIAGED_FILE_C)):
                product_prev, keys_prev = prepareData(sName=UNTRIAGED_FILE_C, otherPercents=0, exactClasses={},
                                                      minComponentDescriptions=1, minProductDescriptions=1)
            dtFrom = ''
            if not os.path.exists(os.path.join(config.DATA_DIR_C, config.UNTRIAGED_FILE_C)):
                dtFrom = '2000-01-01'
            if os.path.exists(downloadUntriaged(dateFrom = dtFrom, baseLogger = logger)):
                product_new, keys_new = prepareData(sName=UNTRIAGED_FILE_C, otherPercents=0, exactClasses={},
                                                    minComponentDescriptions=1, minProductDescriptions=1)
                logger.info('keys_new: ' + str(keys_new))
                
                d1Keys = set(product_prev.keys())
                d2Keys = set(product_new.keys())
                intersectKeys = d1Keys.intersection(d2Keys)
                addedKeys = d2Keys - d1Keys

                product_new2 = {addedKey : product_new[addedKey] for addedKey in addedKeys}
                keys_new2 = {}

                maxValue = 0
                for k, v in keys_prev.items():
                    if v > maxValue:
                        maxValue = v

                for addedKey in addedKeys:
                    maxValue += 1
                    keys_new2[addedKey] = maxValue

                for key in intersectKeys:
                    vp1 = product_prev[key]
                    vp2 = product_new[key]
                    addedprods = []
                    for item in vp2:
                        bExists = False
                        for item1 in vp1:
                            if item1['id'] == item['id']:
                                bExists = True
                                break
                        if not bExists:
                            addedprods.append(item)
                    if len(addedprods) > 0:
                        product_new2[key] = addedprods
                        keys_new2[key] = keys_prev[key]
                    
                product_new = product_new2
                keys_new = keys_new2

                products = [key[:-len(' - Untriaged')] for key in keys_new.keys() 
                            if key[:-len(' - Untriaged')] in untriagedProducts]
                
                logger.info('update products: ' + str(products))
                
                logger.info('update keys_new: ' + str(keys_new))
                logger.info('update product_new.keys: ' + str(product_new.keys()))
                
                logger.info('Update with products count: \n' + str([len(product_new[item + ' - Untriaged']) for item in products]))

                # select best model for new products
                for item in products:
                    # prepare data and select model
                    product_data, keys_data = prepareData(sName='bugDataTest500.csv', sDir=dataDir,
                                                          exactClasses={item: [[]]}, 
                                                          otherPercents = 0,
                                                          minComponentDescriptions=50, minProductDescriptions=50)
                    datasetName = item + config.DATASET_SUFFIX
                    if not os.path.exists(os.path.join(dataDir, datasetName + '.npz')):
                        makeDataset(datasetName=datasetName, data_dict=product_data, productKeys=keys_data, sDir=dataDir)
                        datasetNames = [datasetName]
                        adaptiveModelSelector(logger = logger,
                                              modelNames = modelNames,
                                              datasetNames = datasetNames,
                                              modelsDir = modelsDir,
                                              dataDir = dataDir,
                                              modelsParams = modelsParams)

                    # classify bugs with selected model
                    try:
                        modelName = readFromConfig(cfgModelsSection, datasetName)[0]
                    except Exception as exc1:
                        adaptiveModelSelector(logger = logger,
                                              modelNames = modelNames,
                                              datasetNames = [datasetName],
                                              modelsDir = modelsDir,
                                              dataDir = dataDir,
                                              modelsParams = modelsParams)
                        modelName = readFromConfig(cfgModelsSection, datasetName)[0]
                    logger.info('base model name: ' + modelName)
                    
                    if (not os.path.exists(os.path.join(modelsDir, modelName + '.json')) and 
                        not os.path.exists(os.path.join(modelsDir, modelName + '.pkl')) and
                        not os.path.exists(os.path.join(modelsDir, modelName + '.py'))):
                        logger.warning('Cannot find model: ' + modelName)
                        logger.warning('prediction for ' + item + ' will be not calculated or saved in db')
                        continue
                    logger.info('classify bugs with model: ' + modelName)

                    res = predict_classes(modelName=modelName, testData=product_new[item + ' - Untriaged'],
                                          pKeys=keys_data, pData=product_data)
                    
                    # update db with predicted classes
                    if dbStorage.tryConnect():
                        dbStorage.updateTable(res)
                    else:
                        logger.error('Cannot connect to storage to add bugs info')

            # check changes in bugzilla and update db
            if dbStorage.tryConnect():
                bugs_frame = dbStorage.loadUntriagedBugsFromTable()
                for item in untriagedProducts:
                    checkBugsComponentData(bugs_frame, sBugProduct=item, logger = logger)
                dbStorage.updateTable(bugs_frame, bNewData=False)
            else:
                logger.error('Cannot connect to storage to update bugs info')
            
        except Exception as exc:
            logger.error('Error in BugsDataUpdaterFunc: ' + str(exc))
        finally:            
            writeToConfig(cfgDownloadSection, cfgElementState, cfgValueStateInactive, logger = logger)

    if os.path.exists(modelsDir) and (modelsDir not in sys.path):
        sys.path.append(modelsDir)
    
    while busyState != cfgValueStateStop:
        dtBegin = dt.datetime.now()
        if busyState == cfgValueStateBusy:
            writeToConfig(cfgDownloadSection, cfgElementState, cfgValueStateInactive, logger = logger)
        BugsDataUpdaterFunc()
        if busyState == cfgValueStateStop:
            writeToConfig(cfgDownloadSection, cfgElementState, cfgValueStateInactive, logger = logger)
            break
        timeDelta = (dt.datetime.now() - dtBegin)
        if 0 <= timeDelta.days * 86400 + timeDelta.seconds < updateInterval:
            time.sleep(updateInterval - (timeDelta.days * 86400 + timeDelta.seconds))
        if busyState == cfgValueStateStop:
            writeToConfig(cfgDownloadSection, cfgElementState, cfgValueStateInactive, logger = logger)
            break
            
    logger.info('runBugsDataUpdater finished')