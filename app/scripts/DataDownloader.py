import importlib
import sys
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
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import reduce

import configuration as config
from logger import Logger


# data_handler: function(df, jData)
async def downloadUrls(sUrls, data_frame, data_handler):
    remainingUrls = sUrls
    with ThreadPoolExecutor(max_workers=32) as executor:
        while len(remainingUrls) > 0:
            loop = asyncio.get_event_loop()
            futures = [loop.run_in_executor(executor, requests.get, sUrl)
                       for sUrl in remainingUrls]
            print('futures count = ', len(futures))
            rCount = 0
            remainingUrlsIndices = []
            for response in await asyncio.gather(*futures, return_exceptions=True):
                if isinstance(response, Exception):
                    remainingUrlsIndices.append(rCount)
                else:
                    if len(response.text) > 0:
                        try:
                            jData = response.json()
                            data_handler(data_frame, jData)
                        except Exception as exc:
                            print('Incorrect json response or problem in data_handler. Response = ', response)
                rCount += 1
            print('gathered count = ', rCount - len(remainingUrlsIndices))
            if len(remainingUrlsIndices) > 0:
                gc.collect()
                time.sleep(4)
                remainingUrls = [remainingUrls[i] for i in remainingUrlsIndices]
            else:
                remainingUrls = []


def downloadExtendedData(data_frame):
    def extendedDataHandler(df, jData):
        for sId, obj in jData['bugs'].items():
            df.set_value(int(sId), config.DESCRIPTION, obj['comments'][0]['text'])
    commentUrls = ['https://bugzilla.mozilla.org' + '/rest/bug/' +
                   str(bugId) + '/comment' for bugId in data_frame.bug_id]
    loop = asyncio.get_event_loop()
    loop.run_until_complete(downloadUrls(commentUrls, data_frame, extendedDataHandler))


# provides possibility of saving data after exceptions
dataAll = pd.DataFrame()


def downloadData(resolutions=['FIXED'],
                 sDir=config.DATA_DIR_C, sName=config.DATA_FILENAME_C,
                 nMax=100000,
                 columns=["bug_id", "opendate", "cc_count", "keywords", "longdescs.count", "priority",
                          "classification", "product", "component", "bug_status", "resolution", "short_desc",
                          "rep_platform", "op_sys", "reporter", "version"],
                 # exactProducts and exactComponents will be ignored if components not empty.
                 components={},
                 exactProducts=[],
                 exactComponents=[],
                 dateFrom='2000-01-01', dateTo='Now',
                 baseLogger=None):
    global dataAll
    if baseLogger is None:
        logger = Logger(log_dir=config.LOG_DIR, log_file_base_name='downloadData')
    else:
        logger = baseLogger
    resList = ["---", "FIXED", "INVALID", "WONTFIX", "DUPLICATE", "WORKSFORME", "INCOMPLETE",
               "SUPPORT", "EXPIRED", "MOVED"]
    columnsList = ["bug_id", "opendate", "cc_count", "keywords", "longdescs.count", "priority", "classification",
                   "product", "component", "bug_status", "resolution", "short_desc",
                   "rep_platform", "op_sys", "reporter", "version"]
    if not os.path.isdir(sDir):
        return False
    baseUrl = 'https://bugzilla.mozilla.org/buglist.cgi?query_format=advanced&columnlist='
    bValid = False
    for colName in columns:
        if colName in columnsList:
            if bValid:
                baseUrl += ('%2C' + colName)
            else:
                baseUrl += colName
                bValid = True
    if not bValid:
        return False

    bValid = False
    for res in resolutions:
        if res in resList:
            baseUrl += ('&resolution=' + res)
            bValid = True
    if not bValid:
        return False

    baseUrl += '&ctype=csv'
    if dateFrom is not None and dateTo is not None:
        baseUrl += '&chfield=[Bug%20creation]&chfieldfrom=' + dateFrom + '&chfieldto=' + dateTo

    bResult = False
    fileStarted = False
    retryCount = 5
    retryIndex = 0
    step = 1000
    offset = 0
    downloadedCount = -1
    dataAll = pd.DataFrame()
    sUrls = []
    if len(components) > 0:
        for sProduct, vComponents in components.items():
            for sComponent in vComponents:
                sUrls.append(baseUrl + '&product=' + sProduct + '&component=' + sComponent)
    else:
        if len(exactProducts) > 0:
            if len(exactComponents) > 0:
                for sProduct in exactProducts:
                    for sComponent in exactComponents:
                        sUrls.append(baseUrl + '&product=' + sProduct + '&component=' + sComponent)
            else:
                for sProduct in exactProducts:
                    sUrls.append(baseUrl + '&product=' + sProduct)
        else:
            for sComponent in exactComponents:
                sUrls.append(baseUrl + '&component=' + sComponent)
        if len(exactProducts) == 0 and len(exactComponents) == 0:
            sUrls = [baseUrl]
    #print(len(components), len(sUrls))
    # print(sUrls)
    for sUrl in sUrls:
        logger.info('start loading for: ' + sUrl)
        offset = 0
        downloadedCount = -1
        retryIndex = 0
        while (offset < nMax) and (downloadedCount != 0):
            try:
                sCurrentUrl = sUrl + '&limit=' + \
                    str(min(step, nMax - offset)) + '&offset=' + str(offset)
                logger.info('Start downloading data')
                r = requests.get(sCurrentUrl, allow_redirects=True)
                dataPart = pd.read_csv(io.BytesIO(r.content), low_memory=False)
                dataPart['description'] = ''
                dataPart.set_index(keys='bug_id', drop=False, inplace=True)
                logger.info('Start downloading extended data')
                downloadExtendedData(dataPart)
                logger.info('Finish downloading extended data')
                if dataAll.empty:
                    dataAll = dataPart
                else:
                    dataAll = pd.concat([dataAll, dataPart], ignore_index=True)
                downloadedCount = dataPart.shape[0]
                offset += downloadedCount
                logger.info('downloaded at step: ' + str(downloadedCount))
                logger.info('downloaded all: ' + str(offset))
                dataPart = pd.DataFrame()
                retryIndex = 0
                logger.info('Finish processing data')
                gc.collect()
            except Exception as exc:
                logger.error('error occured: ' + str(exc))
                if offset > 0:
                    if fileStarted:
                        with open(os.path.join(sDir, sName), 'a', encoding='utf-8') as f:
                            dataAll.to_csv(f, header=False, encoding='utf-8')
                            logger.info('append data shape = ' + str(dataAll.shape))
                            dataAll = pd.DataFrame()
                    else:
                        dataAll.to_csv(os.path.join(sDir, sName), encoding='utf-8')
                        logger.info('write data shape = ' + str(dataAll.shape))
                        dataAll = pd.DataFrame()
                        fileStarted = True
                gc.collect()
                if downloadedCount < step:
                    retryIndex += 1
                    time.sleep(61)
                if retryIndex >= retryCount:
                    logger.info('reached max retries count: ' + str(retryCount))
                    break
        if offset > 0:
            if fileStarted:
                with open(os.path.join(sDir, sName), 'a', encoding='utf-8') as f:
                    dataAll.to_csv(f, header=False, encoding='utf-8')
                    logger.info('append data shape = ' + str(dataAll.shape))
                    dataAll = pd.DataFrame()
            else:
                dataAll.to_csv(os.path.join(sDir, sName), encoding='utf-8')
                logger.info('write data shape = ' + str(dataAll.shape))
                dataAll = pd.DataFrame()
                fileStarted = True
            gc.collect()
            logger.info('data loaded for: ' + sUrl)
            bResult = True
    return bResult


# download actual untriaged data, returns path to downloaded file
def downloadUntriaged(dateFrom='', sName=config.UNTRIAGED_FILE_C, baseLogger=None):
    if not downloadData(resolutions=['---'],
                        sDir=config.DATA_DIR_C, sName=sName, nMax=10000,
                        exactComponents=['Untriaged'],
                        dateFrom=dateFrom if '' != dateFrom else (
                            datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d'),
                        dateTo='Now', baseLogger=baseLogger):
        return ''
    else:
        return os.path.join(config.DATA_DIR_C, sName)

# check actual bugs components info on bugzilla


def checkBugsComponentData(data_frame, sBugProduct, logger=None):
    def checkBugsComponentDataHandler(df, jData):
        for obj in jData['bugs']:
            if obj['product'] != sBugProduct:
                df.set_value(int(obj['id']), 'product', obj['product'])
            if obj['component'] != 'Untriaged':
                df.set_value(int(obj['id']), 'component', obj['component'])
    if logger is None:
        print('checkBugsComponentData start')
    else:
        logger.info('checkBugsComponentData start')
    bugsUrls = ['https://bugzilla.mozilla.org' + '/rest/bug/' +
                str(bugId) for bugId in data_frame.bug_id]
    loop = asyncio.get_event_loop()
    loop.run_until_complete(downloadUrls(bugsUrls, data_frame, checkBugsComponentDataHandler))
    if logger is None:
        print('checkBugsComponentData finished')
    else:
        logger.info('checkBugsComponentData finished')
