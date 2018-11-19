import collections
import copy
import os
import re
import sys
import gc
from datetime import datetime, timedelta
from time import strftime
import numpy as np
import scipy as sp
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stopwords
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ParameterGrid
from sklearn.linear_model import SGDClassifier
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

import pickle
from sklearn.externals import joblib

#------------------------
from logger import Logger
import utils
import configuration as config
#------------------------

np.random.seed(config.RANDOM_STATE)


class BugClassifier:
    classifier = None
    classifier_parameters = None
    text_processing_parameters = None
    target_product = None
    components_exclude_list = None
    vectorizer = None
    component2index = None
    index2component = None
    platform_encoder = None
    platform_onehot_encoder = None
    op_sys_encoder = None
    op_sys_onehot_encoder = None
    reporter_encoder = None
    reporter_onehot_encoder = None
    platform_names = None
    op_sys_names = None
    reporter_names = None
    logger = None
    df = None
    x = None
    y = None
    model = None
    stemmer = nltk.stem.porter.PorterStemmer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def __init__(self, logger, classifier=SGDClassifier, classifier_parameters={},
                 text_processing_parameters={},
                 target_product='Firefox',
                 components_exclude_list=['General', 'Untriaged']):
        self.logger = logger
        self.logger.info('Start create bug classifier: {}'.format(
            config.CLASSIFIER_NAMES_DICT[classifier]))
        self.classifier = classifier
        self.classifier_parameters = classifier_parameters
        self.model = self.classifier(**self.classifier_parameters)
        self.text_processing_parameters = text_processing_parameters
        self.target_product = target_product
        self.components_exclude_list = components_exclude_list
        self.logger.info('Finish create bug classifier: {}'.format(
            config.CLASSIFIER_NAMES_DICT[classifier]))

    def save_model(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_file_path = model_dir + '/model_' + \
            config.CLASSIFIER_NAMES_DICT[self.classifier].lower(
            ) + '_' + self.target_product + config.DATASET_SUFFIX + '.pkl'
        self.logger.info('Start save model to file: {}'.format(model_file_path))
        model_dict = {}
        model_dict['classifier'] = self.classifier
        model_dict['classifier_parameters'] = self.classifier_parameters
        model_dict['text_processing_parameters'] = self.text_processing_parameters
        model_dict['target_product'] = self.target_product
        model_dict['components_exclude_list'] = self.components_exclude_list
        model_dict['model'] = self.model
        model_dict['vectorizer'] = self.vectorizer
        model_dict['component2index'] = self.component2index
        model_dict['platform_encoder'] = self.platform_encoder
        model_dict['platform_onehot_encoder'] = self.platform_onehot_encoder
        model_dict['op_sys_encoder'] = self.op_sys_encoder
        model_dict['op_sys_onehot_encoder'] = self.op_sys_onehot_encoder
        model_dict['reporter_encoder'] = self.reporter_encoder
        model_dict['reporter_onehot_encoder'] = self.reporter_onehot_encoder
        joblib.dump(model_dict, model_file_path)
        del model_dict
        gc.collect()
        self.logger.info('Finish save model to file')

    def load_model(self, model_dir, classifier):
        model_file_path = model_dir + '/model_' + \
            config.CLASSIFIER_NAMES_DICT[classifier].lower(
            ) + '_' + self.target_product + config.DATASET_SUFFIX + '.pkl'
        self.logger.info('Start load model from file: {}'.format(model_file_path))
        model_dict = {}
        model_dict = joblib.load(model_file_path)
        self.classifier = model_dict['classifier']
        # TODO: self.classifier != classifier ---> Exception
        self.classifier_parameters = model_dict['classifier_parameters']
        self.model = self.classifier(**self.classifier_parameters)
        self.text_processing_parameters = model_dict['text_processing_parameters']
        self.target_product = model_dict['target_product']
        self.components_exclude_list = model_dict['components_exclude_list']
        self.model = model_dict['model']
        self.vectorizer = model_dict['vectorizer']
        self.component2index = model_dict['component2index']
        self.platform_encoder = model_dict['platform_encoder']
        self.platform_onehot_encoder = model_dict['platform_onehot_encoder']
        self.op_sys_encoder = model_dict['op_sys_encoder']
        self.op_sys_onehot_encoder = model_dict['op_sys_onehot_encoder']
        self.reporter_encoder = model_dict['reporter_encoder']
        self.reporter_onehot_encoder = model_dict['reporter_onehot_encoder']
        self.index2component = {v: k for k, v in self.component2index.items()}
        self.df = None
        self.stemmer = nltk.stem.porter.PorterStemmer()
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.platform_names = np.sort(self.platform_encoder.classes_)
        self.op_sys_names = np.sort(self.op_sys_encoder.classes_)
        self.reporter_names = np.sort(self.reporter_encoder.classes_)
        self.x = None
        self.y = None
        del model_dict
        gc.collect()
        self.logger.info('Finish load model from file')

    def load_bugs_data(self, data_file_path):
        self.logger.info('Start load bugs data from file: {}'.format(data_file_path))
        self.df = pd.read_csv(data_file_path, sep=config.BUGS_DATA_SEPARATOR, low_memory=False,
                              nrows=config.READ_DATA_ROWS_MAX_COUNT)
        if 'Unnamed: 0' in self.df.columns:
            del self.df['Unnamed: 0']
        self.logger.info('Finish load bugs data, rows count: {}'.format(self.df.shape[0]))

    def process_bugs_data(self):
        self.logger.info('Start process bugs data, product: {}'.format(self.target_product))
        df = self.df
        df.fillna(value='', inplace=True)
        df.drop(df.index[~df[config.PRODUCT].isin([self.target_product])], inplace=True)
        df.drop(df.index[df[config.COMPONENT].isin(self.components_exclude_list)], inplace=True)
        gc.collect()

        components = df[config.COMPONENT].value_counts()
        components = components[components >= config.COMPONENT_MIN_BUGS_COUNT]
        components_counter = collections.Counter(components.to_dict())
        components_most_common = components_counter.most_common(config.COMPONENTS_MAX_COUNT)
        components_labels_most_common = []
        for component_label in components_most_common:
            components_labels_most_common.append(component_label[0])
        df.drop(df.index[~df[config.COMPONENT].isin(components_labels_most_common)], inplace=True)
        gc.collect()
        components = df[config.COMPONENT].value_counts()
        components_counter = collections.Counter(components.to_dict())
        components_most_common = components_counter.most_common()
        self.component2index = collections.defaultdict(int)
        for component_index, component_name in enumerate(components_most_common):
            self.component2index[component_name[0]] = component_index
        self.index2component = {v: k for k, v in self.component2index.items()}
        components_count = len(self.component2index)
        self.logger.info('Components count: {}'.format(components_count))

        platform = df[config.PLATFORM].value_counts()
        platform_counter = collections.Counter(platform.to_dict())
        platform_most_common = utils.get_most_common(platform_counter,
                                                     min_count=config.FEATURE_VALUE_MIN_FREQUENCY)
        df.loc[~df[config.PLATFORM].isin(platform_most_common),
               config.PLATFORM] = config.OTHER_PLATFORMS

        opsys = df[config.OP_SYS].value_counts()
        opsys_counter = collections.Counter(opsys.to_dict())
        opsys_most_common = utils.get_most_common(opsys_counter,
                                                  min_count=config.FEATURE_VALUE_MIN_FREQUENCY)
        df.loc[~df[config.OP_SYS].isin(opsys_most_common), config.OP_SYS] = config.OTHER_OP_SYS

        reporter = df[config.REPORTER].value_counts()
        reporter_counter = collections.Counter(reporter.to_dict())
        reporter_most_common = utils.get_most_common(reporter_counter,
                                                     min_count=config.FEATURE_VALUE_MIN_FREQUENCY)
        df.loc[~df[config.REPORTER].isin(reporter_most_common),
               config.REPORTER] = config.OTHER_REPORTERS
        gc.collect()

        bugs_count = df.shape[0]
        self.logger.info('Bugs count: {}'.format(bugs_count))
        self.x = np.empty((bugs_count, config.FEATURES_COUNT), dtype=object)
        self.y = np.empty((bugs_count,), dtype=int)
        x = self.x
        y = self.y
        row_num = 0
        for index, row in df.iterrows():
            x[row_num][config.DESCRIPTION_COLUMN_NUM] = utils.clean_string(
                str(row[config.SUMMARY]) + ' ' + str(row[config.DESCRIPTION])).lower()
            x[row_num][config.PLATFORM_COLUMN_NUM] = str(row[config.PLATFORM])
            x[row_num][config.OP_SYS_COLUMN_NUM] = str(row[config.OP_SYS])
            x[row_num][config.REPORTER_COLUMN_NUM] = str(row[config.REPORTER])
            y[row_num] = self.component2index[row[config.COMPONENT]]
            row_num += 1
        del df
        gc.collect()
        self.logger.info('Finish process bugs data')
        return x, y

    # preprocess bug description (data munging)
    # remove word with len less then min_word_len - option
    # remove digits - option
    # remove stopwords
    # use stemmer or lemmatizer - option
    def __preprocess_text(self, text, min_word_len, remove_digits,
                          use_stemmer_lemmatizer):
        words = [x for x in nltk.word_tokenize(text)]
        new_text = ''
        for word in words:
            if len(word) < min_word_len:
                continue
            if remove_digits == True and word.isdigit() == True:
                continue
            if word in stopwords:
                continue
            if use_stemmer_lemmatizer == config.STEMMER:
                word = self.stemmer.stem(word)
            elif use_stemmer_lemmatizer == config.LEMMATIZER:
                word = self.lemmatizer.lemmatize(word)
            new_text += word + ' '
        return new_text

    def __construct_features(self, x, train_indices, val_indices, column_num,
                             value_others):
        x_train = x[train_indices, column_num]
        x_train = np.append(x_train, value_others)
        x_val = x[val_indices, column_num]
        encoder = LabelEncoder().fit(x_train)
        classes = list(encoder.classes_)
        for index, value in np.ndenumerate(x_val):
            if value not in classes:
                x_val[index] = value_others
        x_train = x_train[:-1]
        x_train = encoder.transform(x_train)
        x_val = encoder.transform(x_val)
        onehot_encoder = OneHotEncoder().fit(x_train[:, np.newaxis])
        x_train = onehot_encoder.transform(x_train[:, np.newaxis])
        x_val = onehot_encoder.transform(x_val[:, np.newaxis])
        return x_train, x_val, encoder, onehot_encoder

    # make train and validation sets for models, converting sourse bugs data
    # to the scipy sparse matrix
    def __make_train_test_sets(self, x, train_indices, val_indices, max_words,
                               word_min_len, remove_digits, use_stemmer_lemmmatizer,
                               ngram_min, ngram_max, min_df, use_idf, tfidf_norm):
        self.logger.info('Start make train and test sets')
        self.logger.info('Train set size: {0}, test set size: {1}'.format(
            len(train_indices), len(val_indices)))
        # SDESCRIPTION

        x_train_description_text = []
        for text in x[train_indices, config.DESCRIPTION_COLUMN_NUM]:
            x_train_description_text.append(self.__preprocess_text(text, word_min_len, remove_digits,
                                                                   use_stemmer_lemmmatizer))
        x_val_description_text = []
        for text in x[val_indices, config.DESCRIPTION_COLUMN_NUM]:
            x_val_description_text.append(self.__preprocess_text(text, word_min_len, remove_digits,
                                                                 use_stemmer_lemmmatizer))

        # Convert bug descriptions (text) to the vectors using tf-idf transformation
        self.vectorizer = TfidfVectorizer(ngram_range=(ngram_min, ngram_max), min_df=min_df,
                                          stop_words=None, lowercase=False, max_features=max_words,
                                          use_idf=use_idf, norm=tfidf_norm).fit(x_train_description_text)

        x_train_description = self.vectorizer.transform(x_train_description_text)
        x_val_description = self.vectorizer.transform(x_val_description_text)
        del x_val_description_text, x_train_description_text
        gc.collect()

        # PLATFORM
        x_train_platform, x_val_platform, self.platform_encoder, self.platform_onehot_encoder = self.__construct_features(x,
                                                                                                                          train_indices, val_indices, config.PLATFORM_COLUMN_NUM,
                                                                                                                          config.OTHER_PLATFORMS)
        # OP_SYS
        x_train_op_sys, x_val_op_sys, self.op_sys_encoder, self.op_sys_onehot_encoder = self.__construct_features(x,
                                                                                                                  train_indices, val_indices, config.OP_SYS_COLUMN_NUM,
                                                                                                                  config.OTHER_OP_SYS)
        # REPORTER
        x_train_reporter, x_val_reporter, self.reporter_encoder, self.reporter_onehot_encoder = self.__construct_features(x,
                                                                                                                          train_indices, val_indices, config.REPORTER_COLUMN_NUM,
                                                                                                                          config.OTHER_REPORTERS)

        self.platform_names = np.sort(self.platform_encoder.classes_)
        self.op_sys_names = np.sort(self.op_sys_encoder.classes_)
        self.reporter_names = np.sort(self.reporter_encoder.classes_)

        x_train = sp.sparse.hstack([x_train_description, x_train_platform,
                                    x_train_op_sys, x_train_reporter]).tocsr()
        x_val = sp.sparse.hstack([x_val_description, x_val_platform,
                                  x_val_op_sys, x_val_reporter]).tocsr()
        del x_train_description, x_train_platform, x_train_op_sys, x_train_reporter
        del x_val_description, x_val_platform, x_val_op_sys, x_val_reporter
        gc.collect()
        self.logger.info('Finish make train and test sets')
        return x_train, x_val

    #
    def make_train_test_sets(self, x, train_indices, test_indices):
        return self.__make_train_test_sets(x, train_indices, test_indices,
                                           self.text_processing_parameters['max_words'],
                                           self.text_processing_parameters['word_min_len'],
                                           self.text_processing_parameters['remove_digits'],
                                           self.text_processing_parameters['use_stemmer_lemmmatizer'],
                                           self.text_processing_parameters['ngram_min'],
                                           self.text_processing_parameters['ngram_max'],
                                           self.text_processing_parameters['min_df'],
                                           self.text_processing_parameters['use_idf'],
                                           self.text_processing_parameters['tfidf_norm'])

    # cross validation of the classifier model
    def __cross_validation(self, model, x, y, cross_validator,
                           text_processing_parameters, cross_validator_executes_count):
        time_start = datetime.now()
        iterations_count = min(cross_validator.get_n_splits(), cross_validator_executes_count)
        iteration_num = 0
        percentage = 0
        self.logger.info('Start cross-validation, iterations count: {}'.format(iterations_count))
        scores = []
        mean_classification_report = None
        split_number = 0
        component_indices = []
        component_names = []
        for index, component_name in self.index2component.items():
            component_indices.append(index)
            component_names.append(component_name)

        for trainval_indices, test_indices in cross_validator.split(x, y):
            iteration_num += 1
            x_trainval, x_test = self.__make_train_test_sets(x, trainval_indices, test_indices,
                                                             text_processing_parameters['max_words'],
                                                             text_processing_parameters['word_min_len'],
                                                             text_processing_parameters['remove_digits'],
                                                             text_processing_parameters['use_stemmer_lemmmatizer'],
                                                             text_processing_parameters['ngram_min'],
                                                             text_processing_parameters['ngram_max'],
                                                             text_processing_parameters['min_df'],
                                                             text_processing_parameters['use_idf'],
                                                             text_processing_parameters['tfidf_norm'])
            # evaluate
            #clf = sklearn.clone(classifier)
            clf = copy.deepcopy(model)
            clf.fit(x_trainval, y[trainval_indices])
            y_pred = clf.predict(x_test)
            score = accuracy_score(y[test_indices], y_pred)
            report = utils.create_classification_report(y, test_indices, y_pred,
                                                        component_indices, component_names)
            if iteration_num > 1:
                mean_classification_report = mean_classification_report.add(report, fill_value=0)
            else:
                mean_classification_report = report
            scores.append(score)
            del clf, x_trainval, x_test, trainval_indices, test_indices
            gc.collect()
            self.logger.info(
                '{0}.Train-test set split, score: {1:.3f}'.format(iteration_num, score))
            percentage = utils.log_progress(
                self.logger, time_start, iteration_num, iterations_count, percentage)
            split_number += 1
            if split_number == cross_validator_executes_count:
                break

        mean_score = np.mean(scores)
        mean_classification_report = mean_classification_report.divide(iterations_count,
                                                                       axis='columns', level=None, fill_value=0)
        mean_classification_report['support'] = mean_classification_report['support'].round().astype(
            'int64')
        support = mean_classification_report['support'].sum(
        ) - mean_classification_report.at['weighted_avg / total', 'support']
        mean_classification_report.at['weighted_avg / total', 'support'] = support
        self.logger.info('Cross-validation score: {:.3f}'.format(mean_score))
        self.logger.info('Cross-validation classification report:\n{}'.format(
            utils.classification_report2string(mean_classification_report)))
        self.logger.info('Finish cross-validation')

        return mean_score, mean_classification_report

    def __find_best_parameters(self, scores_and_best_parameters_list):
        best_parameters = None
        best_score = 0
        for results in scores_and_best_parameters_list:
            if results[0] > best_score:
                best_parameters = results[1]
        return best_parameters

    # classifier hyperparameters grid search with the cross-validation
    def __cross_validation_and_grid_search(self, x, y, classifier,
                                           inner_cv, inner_cv_executes_count,
                                           outer_cv, outer_cv_executes_count,
                                           parameters_grid):

        time_start = datetime.now()
        iterations_count = ((min(inner_cv.get_n_splits(),
                                 inner_cv_executes_count) * len(list(parameters_grid))) + 1) * min(outer_cv.get_n_splits(),
                                                                                                   outer_cv_executes_count)
        iteration_num = 0
        percentage = 0
        self.logger.info(
            'Start cross validation and model parameters grid search, iterations count: {}'.format(iterations_count))
        outer_scores = []
        outer_best_params = []

        outer_split_number = 0
        for trainval_indices, test_indices in outer_cv.split(x, y):
            # find best parameter using inner cross-validation
            best_params = {}
            best_score = -np.inf
            # iterate over parameters
            if inner_cv_executes_count != 0:
                for parameters in parameters_grid:
                    # accumulate score over inner splits
                    cv_scores = []
                    # iterate over inner cross-validation
                    inner_split_number = 0
                    for train_indices, val_indices in inner_cv.split(x[trainval_indices], y[trainval_indices]):
                        inner_split_number += 1

                        x_train, x_val = self.__make_train_test_sets(x, train_indices, val_indices,
                                                                     parameters['max_words'],
                                                                     parameters['word_min_len'],
                                                                     parameters['remove_digits'],
                                                                     parameters['use_stemmer_lemmmatizer'],
                                                                     parameters['ngram_min'],
                                                                     parameters['ngram_max'],
                                                                     parameters['min_df'],
                                                                     parameters['use_idf'],
                                                                     parameters['tfidf_norm'])
                        clf_parameters = copy.deepcopy(parameters)
                        clf_parameters.pop('max_words', None)
                        clf_parameters.pop('word_min_len', None)
                        clf_parameters.pop('remove_digits', None)
                        clf_parameters.pop('use_stemmer_lemmmatizer', None)
                        clf_parameters.pop('ngram_min', None)
                        clf_parameters.pop('ngram_max', None)
                        clf_parameters.pop('min_df', None)
                        clf_parameters.pop('use_idf', None)
                        clf_parameters.pop('tfidf_norm', None)

                        clf = classifier(**clf_parameters)
                        clf.fit(x_train, y[train_indices])
                        # evaluate on inner test set
                        y_pred = clf.predict(x_val)
                        score = accuracy_score(y[val_indices], y_pred)
                        cv_scores.append(score)
                        del clf, x_train, x_val, train_indices, val_indices
                        gc.collect()

                        iteration_num += 1
                        self.logger.info('{0}.iteration, score: {1:.3f}'.format(
                            iteration_num, score), log_to_console=False)
                        percentage = utils.og_progress(
                            logger, time_start, iteration_num, iterations_count, percentage)
                        if inner_split_number == inner_cv_executes_count:
                            break

                    # compute mean score over inner folds
                    mean_score = np.mean(cv_scores)
                    if mean_score > best_score:
                        # if better than so far, remember parameters
                        best_score = mean_score
                        best_params = parameters
                        self.logger.info('{0}.outer train-test split, best current score: {1:.3f}'.format(outer_split_number, score),
                                         log_to_console=False)
                        self.logger.info('{0}.outer train-test split, best model parameters: {1}'.format(outer_split_number, best_params),
                                         log_to_console=False)

            else:
                best_params = parameters_grid[0]

            x_trainval, x_test = self.__make_train_test_sets(x, trainval_indices, test_indices,
                                                             best_params['max_words'],
                                                             best_params['word_min_len'],
                                                             best_params['remove_digits'],
                                                             best_params['use_stemmer_lemmmatizer'],
                                                             best_params['ngram_min'],
                                                             best_params['ngram_max'],
                                                             best_params['min_df'],
                                                             best_params['use_idf'],
                                                             best_params['tfidf_norm'])

            clf_parameters = copy.deepcopy(best_params)
            clf_parameters.pop('max_words', None)
            clf_parameters.pop('word_min_len', None)
            clf_parameters.pop('remove_digits', None)
            clf_parameters.pop('use_stemmer_lemmmatizer', None)
            clf_parameters.pop('ngram_min', None)
            clf_parameters.pop('ngram_max', None)
            clf_parameters.pop('min_df', None)
            clf_parameters.pop('use_idf', None)
            clf_parameters.pop('tfidf_norm', None)
            clf = classifier(**clf_parameters)
            clf.fit(x_trainval, y[trainval_indices])
            # evaluate
            y_pred = clf.predict(x_test)
            score = accuracy_score(y[test_indices], y_pred)
            outer_scores.append(score)
            outer_best_params.append(best_params)
            del clf, x_trainval, x_test, trainval_indices, test_indices
            gc.collect()

            iteration_num += 1
            percentage = utils.log_progress(
                self.logger, time_start, iteration_num, iterations_count, percentage)
            self.logger.info(
                '{0}.outer train-test split, best score: {1:.3f}'.format(outer_split_number, score), log_to_console=False)
            self.logger.info('{0}.outer train-test split, best model parameters: {1}'.format(
                outer_split_number, best_params), log_to_console=False)
            outer_split_number += 1
            if outer_split_number == outer_cv_executes_count:
                break
        results = []
        self.logger.info('Mean score: {:.3f}'.format(np.mean(outer_scores)))
        for idx, score, best_params in zip(range(outer_cv.get_n_splits()), outer_scores, outer_best_params):
            self.logger.info('{0}.outer train-test split, best score: {1:.3f}'.format(idx, score))
            self.logger.info(
                '{0}.outer train-test split, best model parameters: {1}'.format(idx, best_params))
            results.append((score, best_params))
        self.logger.info('Finish cross validation and model parameters grid search')
        return self.__find_best_parameters(results)

    #
    def __train(self, x_train, y, train_indices):
        self.logger.info('Start train model')
        self.model.fit(x_train, y[train_indices])
        self.logger.info('Finish train model')
        gc.collect()
        return self.df

    #
    def test(self, x_test, y, test_indices):
        self.logger.info('Start test model')
        predictions = self.model.predict_proba(x_test)
        pred = np.argmax(predictions, axis=1)
        score = accuracy_score(y[test_indices], pred)
        component_indices = []
        component_names = []
        for index, component_name in self.index2component.items():
            component_indices.append(index)
            component_names.append(component_name)
        report = utils.create_classification_report(y, test_indices, pred,
                                                    component_indices, component_names)
        self.logger.info(
            'Finish test model, score: {:.3f}\nclassification_report:\n{}\n'.format(score, report))
        gc.collect()
        return score, report

    #
    def predict_bug(self, bug):
        # DESCRIPTION
        description = str(bug[config.SUMMARY]) + ' ' + str(bug[config.DESCRIPTION])
        description = self.__preprocess_text(description,
                                             self.text_processing_parameters['word_min_len'],
                                             self.text_processing_parameters['remove_digits'],
                                             self.text_processing_parameters['use_stemmer_lemmmatizer'])
        x_description = self.vectorizer.transform([description])

        # PLATFORM
        platform = str(bug[config.PLATFORM])
        if utils.contain(self.platform_names, platform) == False:
            platform = config.OTHER_PLATFORMS
        x_platform = self.platform_encoder.transform([platform])
        x_platform = self.platform_onehot_encoder.transform(x_platform[:, np.newaxis])

        # OP_SYS
        op_sys = str(bug[config.OP_SYS])
        if utils.contain(self.op_sys_names, op_sys) == False:
            op_sys = config.OTHER_OP_SYS
        x_op_sys = self.op_sys_encoder.transform([op_sys])
        x_op_sys = self.op_sys_onehot_encoder.transform(x_op_sys[:, np.newaxis])

        # REPORTER
        reporter = str(bug[config.REPORTER])
        if utils.contain(self.reporter_names, reporter) == False:
            reporter = config.OTHER_REPORTERS
        x_reporter = self.reporter_encoder.transform([reporter])
        x_reporter = self.reporter_onehot_encoder.transform(x_reporter[:, np.newaxis])

        x = sp.sparse.hstack([x_description, x_platform, x_op_sys, x_reporter]).tocsr()
        probabilities = self.model.predict_proba(x)
        component_index = np.argmax(probabilities, axis=1)[0]
        #component_index = self.model.predict(x)[0]
        component = self.index2component[component_index]
        probability = probabilities[0, component_index]
        return component, probability

    #
    def predict(self, bugs):
        bugs_count = bugs.shape[0]
        self.logger.info('Start bugs prediction, bugs count: {}', format(bugs_count))
        iteration_num = 0
        last_log_percentage = 0
        errors_count = 0
        avg_error_proba = 0
        time_start = datetime.now()
        result = bugs.copy()
        result[config.STORAGE_COLUMN_SUGGESTED_COMPONENT] = ''
        result[config.STORAGE_COLUMN_CONFIDENCE] = 0.0
        for idx, bug in bugs.iterrows():
            suggested_component, confidence = self.predict_bug(bug=bug)
            result.loc[idx, config.STORAGE_COLUMN_SUGGESTED_COMPONENT] = suggested_component
            result.loc[idx, config.STORAGE_COLUMN_CONFIDENCE] = confidence
            if suggested_component != bug[config.COMPONENT]:
                errors_count += 1
                avg_error_proba += confidence
            iteration_num += 1
            last_log_percentage = utils.log_progress(self.logger, time_start, iteration_num, bugs_count,
                                                     last_log_percentage, delta_percentage=10.0)
        self.logger.info('Accuracy: {0:.3f}, avg. error probability: {1:.3f}'.format(1.0 - (errors_count/bugs_count),
                                                                                     avg_error_proba/errors_count))
        self.logger.info('Finish bugs prediction')
        return result

    #
    def train(self, x_train, y, train_indices):
        self.__train(x_train, y, train_indices)

    #
    def load_data_train_test_model(self, data_file_path, test_size):
        self.logger.info('Start load data train and test_model')
        self.load_bugs_data(data_file_path)
        self.process_bugs_data()
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                          random_state=config.RANDOM_STATE)
        train_indices, test_indices = next(splitter.split(self.x, self.y))
        x_train, x_test = self.make_train_test_sets(self.x, train_indices, test_indices)
        self.train(x_train, self.y, train_indices)
        score, class_report = self.test(x_test, self.y, test_indices)
        del train_indices, test_indices, x_train, x_test
        gc.collect()
        self.logger.info('Finish load data train and test_model')
        return score, class_report

    def load_data_and_make_model_cross_validation(self, data_file_path, splits_count):
        self.load_bugs_data(data_file_path)
        self.process_bugs_data()
        cross_validator = StratifiedKFold(splits_count,
                                          shuffle=True, random_state=config.RANDOM_STATE)
        return self.__cross_validation(self.model, self.x, self.y, cross_validator,
                                       text_processing_parameters=self.text_processing_parameters,
                                       cross_validator_executes_count=min(splits_count, config.CROSS_VALIDATION_EXECUTES_COUNT))

    def load_data_and_make_model_parameters_grid_search(self, data_file_path,
                                                        grid_search_splits_count,
                                                        cross_validation_splits_count,
                                                        parameters_grid):
        self.load_bugs_data(data_file_path)
        self.process_bugs_data()
        inner_cross_validator = StratifiedKFold(grid_search_splits_count,
                                                shuffle=True, random_state=config.RANDOM_STATE)
        outer_cross_validator = StratifiedKFold(cross_validation_splits_count,
                                                shuffle=True, random_state=config.RANDOM_STATE)
        return self.__cross_validation_and_grid_search(self.x, self.y, self.classifier,
                                                       inner_cross_validator,
                                                       min(grid_search_splits_count,
                                                           config.GRID_SEARCH_EXECUTES_COUNT),
                                                       outer_cross_validator,
                                                       min(cross_validation_splits_count,
                                                           config.CROSS_VALIDATION_EXECUTES_COUNT),
                                                       ParameterGrid(parameters_grid))
