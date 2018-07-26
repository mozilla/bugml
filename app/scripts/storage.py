import mysqlx
import base64
import pandas as pd
from datetime import datetime
import time

#------------------------
from logger import Logger
import utils
import configuration as config
#-----------------------


class Storage:
    host = None
    port = None
    user = None
    password = None
    charset = None
    collation = None
    database_name = None
    tables = None
    logger = None

    def __init__(self, logger,
                 host=config.STORAGE_HOST,
                 port=config.STORAGE_PORT,
                 user=config.STORAGE_USER,
                 password=config.STORAGE_PASSWORD,
                 charset=config.STORAGE_CHARSET,
                 collation=config.STORAGE_CHARSET_COLLATION,
                 database_name=config.STORAGE_DATABASE_NAME,
                 tables={config.BUGS_TABLE_NAME_C: config.BUGS_TABLE_COLUMNS_C}
                 ):
        self.logger = logger
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.charset = charset
        self.collation = collation
        self.database_name = database_name
        self.tables = tables

    def __del__(self):
        pass
#        self.disconnect()

    @staticmethod
    def sqlWrapper(s, c='`'):
        return c + s.replace(c, '') + c if c != '' else s

    @staticmethod
    def makeTableCreator(tableName, tableColumns, primaryKey=[]):
        res = 'CREATE TABLE IF NOT EXISTS ' + Storage.sqlWrapper(tableName) + ' ('
        for name, coltype in tableColumns.items():
            res += '\n' + name + ' ' + coltype + ','
        if len(primaryKey) > 0:
            res += '\nPRIMARY KEY (' + ', '.join(primaryKey) + ')'
        res += '\n) '
        return res

    @staticmethod
    def makeTableInserter(tableName, columnsValues):
        res = 'INSERT INTO ' + Storage.sqlWrapper(tableName) + ' '
        if isinstance(columnsValues, dict):
            cols = '(' + ','.join(columnsValues.keys()) + ')'
            values = 'VALUES (' + ', '.join(columnsValues.values()) + ')'
            res += cols + '\n ' + values
        else:
            if isinstance(columnsValues, list):
                res += '\n VALUES (' + ', '.join(columnsValues) + ')'
            else:
                res += '\n VALUES (' + str(columnsValues) + ')'
        return res

    @staticmethod
    def makeTableUpdater(tableName, whereValues, columnsValues):
        res = 'UPDATE ' + Storage.sqlWrapper(tableName) + ' '
        if isinstance(columnsValues, dict) and isinstance(whereValues, dict) and len(columnsValues) > 0:
            sets = 'SET ' + ','.join([key + ' = ' + value for key, value in columnsValues.items()])
            if len(whereValues) > 0:
                wheres = ' WHERE ' + \
                    ' AND '.join([key + ' = ' + value for key, value in whereValues.items()])
            res += sets + '\n ' + wheres
        else:
            raise AssertionError(
                'makeTableUpdater require dict type for whereValues and columnsValues')
        return res

    @staticmethod
    def encode64(s):
        return base64.encodebytes(s.encode('utf-8')).decode('utf-8')

    @staticmethod
    def decode64(s):
        return base64.decodebytes(s.encode('utf-8')).decode('utf-8')

    @staticmethod
    def getSession(host, port, user, password):
        return mysqlx.get_session({'user': user, 'password': password,
                                   'host': host, 'port': port,
                                   'auth': mysqlx.Auth.PLAIN})

    @staticmethod
    def closeSession(session):
        if session is not None:
            if session.is_open():
                session.close()

    def startSession(self):
        return self.getSession(self.host, self.port, self.user, self.password)

    def tryConnect(self, retryCount=config.STORAGE_CONNECT_RETRY_COUNT, timeInterval=config.STORAGE_CONNECT_INTERVAL):
        index = 0
        session = None
        while session is None and index < retryCount:
            try:
                session = self.startSession()
            except Exception as exc:
                self.closeSession(session)
                session = None
            finally:
                index += 1
                if session is not None:
                    self.closeSession(session)
                    return True
                else:
                    time.sleep(timeInterval)
        self.logger.warning('Cannot connect to db.')
        return False

    # create database and add selected tables to db if they are not already exists.
    def configureDB(self):
        session = self.startSession()
        try:
            res = session.sql('SHOW DATABASES').execute()
            dbNames = [item.get_string('Database').lower() for item in res.fetch_all()]
            self.logger.info(str(dbNames))
            if self.database_name.lower() not in dbNames:
                self.logger.info(self.database_name + ': creating... ')
                session.sql('CREATE DATABASE ' + self.sqlWrapper(self.database_name)).execute()
                session.sql('USE ' + self.sqlWrapper(self.database_name)).execute()
                for name, cols in self.tables.items():
                    req = self.makeTableCreator(name, cols, primaryKey=[config.STORAGE_COLUMN_ID])
                    session.sql(req).execute()
            else:
                self.logger.info(self.database_name.replace('\'', '') + ' exists.')
                session.sql('USE ' + self.sqlWrapper(self.database_name)).execute()
                dbs1 = session.get_schema(self.database_name.replace('\'', ''))
                for name, cols in self.tables.items():
                    if name not in [table.get_name() for table in dbs1.get_tables()]:
                        req = self.makeTableCreator(name, cols, primaryKey=[
                                                    config.STORAGE_COLUMN_ID])
                        session.sql(req).execute()
        except BaseException as exc:
            self.logger.error('Error in configureDB: ' + str(exc))
        finally:
            self.closeSession(session)

    # Insert new data or update existing data with new values. Input data_fame differs for these scenarios.
    def updateTable(self, data_frame, bNewData=True, tableName=config.BUGS_TABLE_NAME_C):
        session = self.startSession()
        try:
            dbs1 = session.get_schema(self.sqlWrapper(self.database_name, ''))
            if not dbs1.exists_in_database():
                self.configureDB()
                dbs1 = session.get_schema(self.database_name.replace('\'', ''))

            session.sql('USE ' + self.sqlWrapper(self.database_name.replace('\'', ''))).execute()
            self.logger.info('updateTable: with data_frame \n' + str(data_frame))
            if bNewData:
                for index, row in data_frame.iterrows():
                    if not isinstance(index, int):
                        self.logger.warning('updateTable: data_frame index must be integer')
                        break
                    data = row['data'].split(' ||| ')
                    label = row['classes']
                    dates = row['date']
                    columnsValues = {
                        self.sqlWrapper(config.STORAGE_COLUMN_ID): str(index),
                        self.sqlWrapper(config.STORAGE_COLUMN_OPENDATE):
                            'STR_TO_DATE(' + self.sqlWrapper(dates, '\'') +
                        ', \'%Y-%m-%d %H:%i:%s\')',
                        self.sqlWrapper(config.STORAGE_COLUMN_PREDICTIONS): self.sqlWrapper(label, '\''),
                        self.sqlWrapper(config.STORAGE_COLUMN_REPORTER): self.sqlWrapper(data[0], '\''),
                        self.sqlWrapper(config.STORAGE_COLUMN_SUMMARY): self.sqlWrapper(self.encode64(data[1]), '\''),
                        self.sqlWrapper(config.STORAGE_COLUMN_OP_SYS): self.sqlWrapper(data[2], '\''),
                        self.sqlWrapper(config.STORAGE_COLUMN_PLATFORM): self.sqlWrapper(data[3], '\''),
                        self.sqlWrapper(config.STORAGE_COLUMN_DESCRIPTION): self.sqlWrapper(self.encode64(data[4]), '\'')
                    }
                    session.sql(self.makeTableInserter(
                        tableName, columnsValues=columnsValues)).execute()
            else:
                for index, row in data_frame.iterrows():
                    if not isinstance(index, int):
                        self.logger.warning('updateTable: data_frame index must be integer')
                        break
                    columnsValues = {}
                    if row[config.PRODUCT] is not None and len(row[config.PRODUCT]) > 0:
                        columnsValues[self.sqlWrapper(config.STORAGE_COLUMN_PRODUCT)] = self.sqlWrapper(
                            row[config.PRODUCT], '\'')
                    if row[config.COMPONENT] is not None and len(row[config.COMPONENT]) > 0:
                        columnsValues[self.sqlWrapper(config.STORAGE_COLUMN_COMPONENT)] = self.sqlWrapper(
                            row[config.COMPONENT], '\'')
                    if len(columnsValues) > 0:
                        session.sql(self.makeTableUpdater(tableName, {config.STORAGE_COLUMN_ID: str(index)},
                                                          columnsValues=columnsValues)).execute()

            self.closeSession(session)
            return True
        except BaseException as exc:
            self.logger.error('Error in updateTable: ' + str(exc))
            self.closeSession(session)
            return False

    def selectBugsFromTable(self, tableName=config.BUGS_TABLE_NAME_C):
        res = pd.DataFrame()
        session = self.startSession()
        try:
            session.sql('USE ' + self.sqlWrapper(self.database_name)).execute()
            queryResult = session.sql('SELECT * FROM ' + tableName).execute()
            self.logger.info(str([int(item.get_string(config.STORAGE_COLUMN_ID))
                                  for item in queryResult.fetch_all()]))
        except BaseException as exc:
            self.logger.error('Error in loadUntriagedBugsFromTable: ' + str(exc))
        finally:
            self.closeSession(session)

    # return DataFrame with array of bug id (for checking and updating state of them)
    def loadUntriagedBugsFromTable(self, tableName=config.BUGS_TABLE_NAME_C):
        res = pd.DataFrame()
        session = self.startSession()
        try:
            session.sql('USE ' + self.sqlWrapper(self.database_name)).execute()
            queryResult = session.sql('SELECT ' + config.STORAGE_COLUMN_ID +
                                      ' FROM ' + tableName +
                                      ' WHERE ' + config.STORAGE_COLUMN_COMPONENT + ' IS NULL').execute()
            res[config.STORAGE_COLUMN_ID] = [
                int(item.get_string(config.STORAGE_COLUMN_ID)) for item in queryResult.fetch_all()]
        except BaseException as exc:
            self.logger.error('Error in loadUntriagedBugsFromTable: ' + str(exc))
        finally:
            self.closeSession(session)

        res[config.COMPONENT] = None
        res[config.PRODUCT] = None
        res.set_index(config.STORAGE_COLUMN_ID, drop=False, inplace=True)
        return res

    # returns list of objects with bug_id and processed bugs predictions (3 by default)
    # example:
    # [{ 'bug_id' : '1',
    #   'short_desc' : 'some short description of bug',
    #   'predictions' : { 'Firefox - General' : 0.78, 'Firefox - Installer' : 0.15, 'Firefox - Sync' : 0.05}
    # }]
    # for API
    def loadPredictionsForListFromTable(self, bugsList, tableName=config.BUGS_TABLE_NAME_C):
        if not isinstance(bugsList, list) or len(bugsList) <= 0:
            self.logger.error(
                'loadBugsPredictionsFromDB requires type list for bugsList and not empty bugsList')
            raise AssertionError(
                'loadBugsPredictionsFromDB requires type list for bugsList and not empty bugsList')
        session = self.startSession()
        res = []
        try:
            session.sql('USE ' + self.sqlWrapper(self.database_name)).execute()
            for bugId in bugsList:
                if not isinstance(bugId, int):
                    self.logger.warning(
                        'loadPredictionsForListFromDB: bugsList must contains only integer numbers')
                    break
                queryResult = session.sql('SELECT ' +
                                          ', '.join([config.STORAGE_COLUMN_ID,
                                                     config.STORAGE_COLUMN_SUMMARY, config.STORAGE_COLUMN_PREDICTIONS]) +
                                          ' FROM ' + self.sqlWrapper(tableName) +
                                          ' WHERE ' + config.STORAGE_COLUMN_ID + ' = ' + str(bugId)).execute()
                results = queryResult.fetch_all()
                if queryResult.count > 0:
                    predList = results[0].get_string(config.STORAGE_COLUMN_PREDICTIONS).split(', ')
                    res.append({
                        config.STORAGE_COLUMN_ID: bugId,
                        config.STORAGE_COLUMN_SUMMARY: self.decode64(results[0].get_string(config.STORAGE_COLUMN_SUMMARY)),
                        config.STORAGE_COLUMN_PREDICTIONS: {item[:item.rfind('(')]: item[item.rfind('(')+1: item.rfind(')')]
                                                            for item in predList}
                    })
        except BaseException as exc:
            self.logger.error('Error in loadPredictionsForListFromTable:' + str(exc))
        finally:
            self.closeSession(session)
        return res

    # returns list with objects with bug_id, short_desc and processed bugs predictions (3 by default)
    # example:
    # [{ 'bug_id' : '1',
    #   'short_desc' : 'some short description of bug',
    #   'predictions' : { 'Firefox - General' : 0.78, 'Firefox - Installer' : 0.15, 'Firefox - Sync' : 0.05}
    # }]
    # for API
    def loadPredictionsForDateFromTable(self, dateStart, dateEnd, tableName=config.BUGS_TABLE_NAME_C):
        res = []
        session = self.startSession()
        try:
            session.sql('USE ' + self.sqlWrapper(self.database_name)).execute()
            self.logger.info(str(type(dateStart)) + '    ' + str(type(dateEnd)))
            sDateStart = dateStart
            if not isinstance(dateStart, str):
                sDateStart = dateStart.strftime('%Y-%m-%d')
            sDateEnd = dateEnd
            if not isinstance(dateEnd, str):
                sDateEnd = dateEnd.strftime('%Y-%m-%d')
            queryResult = session.sql('SELECT ' +
                                      ', '.join([config.STORAGE_COLUMN_ID, config.STORAGE_COLUMN_OPENDATE,
                                                 config.STORAGE_COLUMN_SUMMARY, config.STORAGE_COLUMN_PREDICTIONS]) +
                                      ' FROM ' + self.sqlWrapper(tableName) +
                                      ' WHERE ' + config.STORAGE_COLUMN_OPENDATE +
                                      ' BETWEEN STR_TO_DATE(' + self.sqlWrapper(sDateStart, '\'') +
                                      ', \'%Y-%m-%d %H:%i:%s\') AND STR_TO_DATE(' + self.sqlWrapper(sDateEnd, '\'') +
                                      ', \'%Y-%m-%d %H:%i:%s\')').execute()
            results = queryResult.fetch_all()
            for resItem in results:
                predList = resItem.get_string(config.STORAGE_COLUMN_PREDICTIONS).split(', ')
                res.append(
                    {
                        config.STORAGE_COLUMN_ID: int(resItem.get_string(config.STORAGE_COLUMN_ID)),
                        config.STORAGE_COLUMN_SUMMARY: self.decode64(resItem.get_string(config.STORAGE_COLUMN_SUMMARY)),
                        config.STORAGE_COLUMN_PREDICTIONS: {item[:item.rfind('(')]: item[item.rfind('(')+1: item.rfind(')')]
                                                            for item in predList}
                    })
        except BaseException as exc:
            self.logger.error('Error in loadPredictionsForDateFromTable:' + str(exc))
        finally:
            self.closeSession(session)
        return res
