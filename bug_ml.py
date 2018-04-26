'''
1. Description
Machine learning script for estimate the accuracy of the bugzilla bugs classifying
using different algorithms: 
 1) Naive Bayes
 2) Logistic Regression
 3) Support Vector Machine with Core
 4) Random Forest
 5) K-Nearest-Neighbors
Bug descriptions (text) converted to the vectors using tf-idf transformation
Other bug features (reporter, platform, OS) processed as category data
Input data for the script - csv file with examples of bugs
(including bugs features - short description (summary), description (comments),
reporter, reporter platform and OS)
Output script results:
 1) Сlassification models accuracy
 2) Best models hyperparameters
2. Prerequisites
 1) Launguage: Python 3.6
 2) Python packages: numpy, scipy, pandas, sklearn, nltk.
 3) Python Scientific Development Environment: Spyder, Jupyter notebook or another.
3. How to use
Find file "bug_ml_tfidf.py".
Open it with Spyder or Jupyter notebook.
Install required python packages if they are not already installed.
Launch script.
'''

import collections, copy, os, re, sys
from datetime import datetime, timedelta
from time import strftime
import numpy as np
import scipy as sp
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# path to csv file with bugs features (desription, reporter, etc)
INPUT_DATA_FILE_PATH = './data/bugs_100k.csv'
# 
DATA_SEP = ','

LOG_DIR_PATH = './logs_bug_ml_tfidf'

if not os.path.exists(LOG_DIR_PATH):
    os.makedirs(LOG_DIR_PATH)

# path to the bugs processing log
LOG_FILE_PATH = LOG_DIR_PATH + '/log__' + datetime.now().strftime('%Y_%m_%d__%H_%M_%S') + '.txt'

# choose classifier to train
TRAIN_NB  = True # Naive Bayes
TRAIN_LR  = True # Logistic Regression
TRAIN_SVC = True # Support Vector Machine with Core
TRAIN_RF  = True # Random Forest
TRAIN_KNN = True # K-Nearest-Neighbors

# max count of bug examples reading from file
MAX_COUNT_DATA_ROWS = 25000

# max count of words in vocabulary
MAX_WORDS = 50000

# max count of classes processing in train-test set
# to the most common claases belong classes with
# top count of bags
MOST_COMMON_CLASSES_COUNT_LIST = [50, 20, 10, 5, 3, 2]

# if this flag true, then "virtual" class will be create
# in this virtual class will be included bugs of the classes,
# which not belong to the most common classes
CREATE_VIRTUAL_CLASS_OTHER = True

OTHER_CLASSES = 'OtherClasses'

# the min repeat count of feature (reporter, platform, OS) value in feature 
MIN_TIMES_COUNT_IN_FEATURE = 5

# count of splits in cross-validation and grid search models hyperparams
INNER_CV_SPLITS_COUNT = 4
INNER_CV_EXECUTES_COUNT = 0
OUTER_CV_SPLITS_COUNT = 4
OUTER_CV_EXECUTES_COUNT = 4

# if flag true - then concrete bugs feature will be processed 
USE_PLATFORM = True
USE_OP_SYS = True
USE_REPORTER = True
USE_DESCRIPTION = True

# text preprocessing options
NOT = 'not'
STEMMER = 'stemmer'
LEMMATIZER = 'lemmatizer'

# feature names in input csv file
PRODUCT = 'product'
COMPONENT = 'component'
SUMMARY = 'short_desc'
PLATFORM = 'rep_platform'
OP_SYS = 'op_sys'
REPORTER = 'reporter'
DESCRIPTION = 'description'

# main feature of the bugs, by which the bugs will be classified 
# set COMPONENT or PRODUCT
CLASS_LABEL = COMPONENT

# real feature values   rename to these virtual feature values
OTHER_PLATFORMS = 'OtherPlatforms'
OTHER_OS = 'OtherOS'  
OTHER_REPORTERS = 'OtherReporters'    

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

f = open('file.txt', 'w')
LOG_FILE = open(LOG_FILE_PATH, 'w')

DF_BASE = pd.read_csv(INPUT_DATA_FILE_PATH, sep=DATA_SEP, low_memory=False, 
                      nrows=MAX_COUNT_DATA_ROWS)

# stemmer, lemmatizer and stopwords using tp preprocess bug description
PORTER_STEMMER = nltk.stem.porter.PorterStemmer()
WORD_NET_LEMMATIZER = nltk.stem.WordNetLemmatizer()
STOPWORDS = set(nltk.corpus.stopwords.words('english'))

# log message to console output and to the log file
def log_message(message, log_console=True, close_log_file=False):
    global LOG_FILE
    LOG_FILE.write(message + '\n')
    LOG_FILE.flush()
    if log_console == True:
        print(message)
    if close_log_file == True:
        LOG_FILE.close()

# help functions for convert time to string
def time2str(hours, minutes, seconds):
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(hours, minutes, seconds)

def timetotal2str(time_passed, iteration_num, iterations_count):
    seconds_passed = time_passed.total_seconds()                
    seconds_total =  (iterations_count * seconds_passed) / iteration_num               
    hours = seconds_total // 3600                 
    minutes = (seconds_total % 3600) // 60
    seconds = seconds_total - (hours * 3600 + minutes * 60)
    return time2str(hours, minutes, seconds)
    
def timedelta2str(time_delta):    
    total_seconds = time_delta.total_seconds()                               
    hours = total_seconds // 3600                 
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds - (hours * 3600 + minutes * 60)
    return time2str(hours, minutes, seconds)    

# calculate computing progress and log it
def log_computing_progress(time_start, iteration_num, iterations_count, current_percentage):
    time_passed = datetime.now() - time_start                
    new_percentage = 100.0 * iteration_num / iterations_count
    if new_percentage >= current_percentage + 1.0:
        current_percentage = new_percentage
        log_message('processed: {:>5.1f}%, time passed: {}, total time est.: {}'.format(current_percentage,
              timedelta2str(time_passed),timetotal2str(time_passed, iteration_num, iterations_count)))

# remove useless symbols from bug description and convert the rest to lower case (data munging) 
def clean_lower_string(string):
    string = re.sub(r'[^A-Za-z0-9()\'\`]', ' ', string)   
    return string.strip().lower()

# preprocess bug description (data munging)
# remove word with len less then min_word_len - option
# remove digits - option
# remove stopwords
# use stemmer or lemmatizer - option
def preprocess_text(text, min_word_len, remove_digits, use_stemmer_lemmatizer):    
    global STEMMER
    global LEMMATIZER
    global STOPWORDS    
    global PORTER_STEMMER
    global WORD_NET_LEMMATIZER
    
    words = [x for x in nltk.word_tokenize(text)]
    new_text = ''
    for word in words:
        if len(word) < min_word_len:
            continue
        if remove_digits == True and word.isdigit() == True:
            continue    
        if word in STOPWORDS:
            continue
        if use_stemmer_lemmatizer == STEMMER:
            word = PORTER_STEMMER.stem(word)
        elif use_stemmer_lemmatizer == LEMMATIZER:
            word = WORD_NET_LEMMATIZER.lemmatize(word)                                       
        new_text += word + ' '
    return new_text

# make train and validation sets for models, converting sourse bugs data
# to the scipy sparse matrix (sparse_output=True) or numpy 2d array
def make_train_validation_sets(x, train_indices, val_indices, max_words, 
                               word_min_len, remove_digits, use_stemmer_lemmmatizer,
                               ngram_min, ngram_max, min_df, use_idf, tfidf_norm, 
                               sparse_output=True):    
    global USE_PLATFORM
    global OTHER_PLATFORMS
    global USE_OP_SYS
    global OTHER_OS
    global USE_REPORTER
    global OTHER_REPORTERS
    
    column_num = 0
    train_stack = []
    val_stack = []
                                    
    x_train_summary_text = []
    for text in x[train_indices, column_num]:
        x_train_summary_text.append(preprocess_text(text, word_min_len, remove_digits, 
                                                    use_stemmer_lemmmatizer))            
    
    x_val_summary_text = []
    for text in x[val_indices, column_num]:            
        x_val_summary_text.append(preprocess_text(text, word_min_len, remove_digits, 
                                                  use_stemmer_lemmmatizer))                 
    
    # Convert bug descriptions (text) to the vectors using tf-idf transformation
    summary_vectorizer = TfidfVectorizer(ngram_range=(ngram_min, ngram_max), min_df=min_df,
                                         stop_words=None, lowercase=False, max_features=max_words,
                                         use_idf=use_idf, norm=tfidf_norm).fit(x_train_summary_text)
    
    x_train_summary = summary_vectorizer.transform(x_train_summary_text)
    x_val_summary = summary_vectorizer.transform(x_val_summary_text)       
    if sparse_output==False:
        x_train_summary = x_train_summary.toarray()
        x_val_summary = x_val_summary.toarray()        
    train_stack.append(x_train_summary)
    val_stack.append(x_val_summary)
    column_num += 1
    
    if USE_PLATFORM == True:
        x_train_platform = x[train_indices, column_num]
        x_train_platform = np.append(x_train_platform, OTHER_PLATFORMS)
        x_val_platform = x[val_indices, column_num]                    
        platform_encoder = LabelEncoder().fit(x_train_platform)
        platform_classes = list(platform_encoder.classes_)
        for index, platform in np.ndenumerate(x_val_platform):
            if platform not in platform_classes:
                x_val_platform[index] =  OTHER_PLATFORMS 
        x_train_platform = x_train_platform[:-1]
        x_train_platform = platform_encoder.transform(x_train_platform)
        x_val_platform = platform_encoder.transform(x_val_platform) 
        platform_onehot_encoder = OneHotEncoder().fit(x_train_platform[:, np.newaxis])
        
        x_train_platform = platform_onehot_encoder.transform(x_train_platform[:, np.newaxis])
        x_val_platform = platform_onehot_encoder.transform(x_val_platform[:, np.newaxis])
        if sparse_output==False:
            x_train_platform = x_train_platform.toarray()
            x_val_platform = x_val_platform.toarray()
        
        train_stack.append(x_train_platform)
        val_stack.append(x_val_platform)                    
        column_num += 1
    
    if USE_OP_SYS == True:
        x_train_os = x[train_indices, column_num]
        x_train_os = np.append(x_train_os, OTHER_OS)
        x_val_os = x[val_indices, column_num]                    
        os_encoder = LabelEncoder().fit(x_train_os)
        os_classes = list(os_encoder.classes_)
        for index, opsys in np.ndenumerate(x_val_os):
            if opsys not in os_classes:
                x_val_os[index] = OTHER_OS
                
        x_train_os = x_train_os[:-1]
        x_train_os = os_encoder.transform(x_train_os)
        x_val_os = os_encoder.transform(x_val_os)  
        os_onehot_encoder = OneHotEncoder().fit(x_train_os[:, np.newaxis])
        
        x_train_os = os_onehot_encoder.transform(x_train_os[:, np.newaxis])
        x_val_os = os_onehot_encoder.transform(x_val_os[:, np.newaxis])
        if sparse_output==False:
            x_train_os = x_train_os.toarray()
            x_val_os = x_val_os.toarray()        
        
        train_stack.append(x_train_os)
        val_stack.append(x_val_os)                     
        column_num += 1
    
    if USE_REPORTER == True:
        x_train_reporter = x[train_indices, column_num]
        x_train_reporter = np.append(x_train_reporter, OTHER_REPORTERS)
        x_val_reporter = x[val_indices, column_num]                                                                                
        reporter_encoder = LabelEncoder().fit(x_train_reporter)
        reporter_classes = list(reporter_encoder.classes_)
        for index, reporter in np.ndenumerate(x_val_reporter):                
            if reporter not in reporter_classes:
                x_val_reporter[index] =  OTHER_REPORTERS 
        x_train_reporter = x_train_reporter[:-1]
        x_train_reporter = reporter_encoder.transform(x_train_reporter)
        x_val_reporter = reporter_encoder.transform(x_val_reporter)
        reporter_onehot_encoder = OneHotEncoder().fit(x_train_reporter[:, np.newaxis])
        
        x_train_reporter = reporter_onehot_encoder.transform(x_train_reporter[:, np.newaxis])
        x_val_reporter = reporter_onehot_encoder.transform(x_val_reporter[:, np.newaxis])
        if sparse_output==False:
            x_train_reporter = x_train_reporter.toarray()
            x_val_reporter = x_val_reporter.toarray()
    
        train_stack.append(x_train_reporter)
        val_stack.append(x_val_reporter)   
    
    if sparse_output==True:
        x_train = sp.sparse.hstack(train_stack)    
        x_val = sp.sparse.hstack(val_stack)
    else:
        x_train = np.hstack(train_stack)
        x_val = np.hstack(val_stack)  
        
    return x_train, x_val

# cross validation of the classifier model
def cross_validation(classifier, x, y, cv, preprocess_parameters, sparse_data=True):    
    time_start = datetime.now()
    iterations_count = cv.get_n_splits()
    iteration_num = 0
    percentage = 0
    log_message('Start cross_validation, iterations count: {}, time: {}'.format(iterations_count, 
                datetime.now().strftime('%H:%M:%S')))
    scores = []            
    for trainval_indices, test_indices in cv.split(x, y):  
        iteration_num += 1                    
        x_trainval, x_test = make_train_validation_sets(x, trainval_indices, test_indices,
                                                    preprocess_parameters['max_words'],
                                                    preprocess_parameters['word_min_len'],
                                                    preprocess_parameters['remove_digits'],
                                                    preprocess_parameters['use_stemmer_lemmmatizer'],
                                                    preprocess_parameters['ngram_min'],
                                                    preprocess_parameters['ngram_max'],
                                                    preprocess_parameters['min_df'],
                                                    preprocess_parameters['use_idf'],
                                                    preprocess_parameters['tfidf_norm'],
                                                    sparse_data)                                                    
        # evaluate                
        classifier.fit(x_trainval, y[trainval_indices])
        y_pred = classifier.predict(x_test)
        score = accuracy_score(y[test_indices], y_pred)                                                        
        scores.append(score)
        log_message('{0}.train-test set split, score: {1:.3f}'.format(iteration_num, score))        
        log_computing_progress(time_start, iteration_num, iterations_count, percentage)
                        
    mean_score = np.mean(scores)
    log_message('mean score: {:.3f}'.format(mean_score))
    log_message('Finish cross_validation, time duration: {}'.format(timedelta2str(datetime.now() - time_start)))
                
    return mean_score

# classifier hyperparameters grid search with the cross-validation
def cross_validation_and_grid_search(x, y, classifier,
              inner_cv, inner_cv_executes_count,
              outer_cv, outer_cv_executes_count, 
              parameter_grid, classes_count, sparse_data=True):
        
    time_start = datetime.now()
    iterations_count = ((min(inner_cv.get_n_splits(), 
                             inner_cv_executes_count) * len(list(parameter_grid))) + 1) * min(outer_cv.get_n_splits(),
                             outer_cv_executes_count)
    iteration_num = 0
    percentage = 0
    log_message('Start cross_validation_and_grid_search, iterations count: {}, time: {}'.format(iterations_count,
          datetime.now().strftime('%H:%M:%S')))
    outer_scores = []
    outer_best_params = []

    outer_split_number = 0
    for trainval_indices, test_indices in outer_cv.split(x, y):              
        # find best parameter using inner cross-validation
        best_params = {}
        best_score = -np.inf
        # iterate over parameters
        if inner_cv_executes_count != 0:
            for parameters in parameter_grid:
                # accumulate score over inner splits
                cv_scores = []
                # iterate over inner cross-validation
                inner_split_number = 0
                for train_indices, val_indices in inner_cv.split(x[trainval_indices], y[trainval_indices]): 
                    inner_split_number += 1
                    
                    x_train, x_val = make_train_validation_sets(x, train_indices, val_indices,
                                                                parameters['max_words'],
                                                                parameters['word_min_len'],
                                                                parameters['remove_digits'],
                                                                parameters['use_stemmer_lemmmatizer'],              
                                                                parameters['ngram_min'],
                                                                parameters['ngram_max'],
                                                                parameters['min_df'],
                                                                parameters['use_idf'],
                                                                parameters['tfidf_norm'],
                                                                sparse_data)                                
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
                    del clf
                    
                    iteration_num += 1
                    log_message('{0}.iteration, score: {1:.3f}'.format(iteration_num, score), log_console=False)  
                    log_computing_progress(time_start, iteration_num, iterations_count, percentage)                     
                    if inner_split_number == inner_cv_executes_count:
                        break
                
                # compute mean score over inner folds
                mean_score = np.mean(cv_scores)
                if mean_score > best_score:
                    # if better than so far, remember parameters
                    best_score = mean_score
                    best_params = parameters                    
                    log_message('{0}.outer train-test split, BEST CURRENT SCORE: {1:.3f}'.format(outer_split_number, score), 
                                log_console=False)
                    log_message('{0}.outer train-test split, BEST MODEL HYPERPARAMETERS: {1}'.format(outer_split_number, best_params), 
                                log_console=False)                    
                    
        else:
            best_params = parameter_grid[0]
        
        x_trainval, x_test = make_train_validation_sets(x, trainval_indices, test_indices,
                                                    best_params['max_words'],
                                                    best_params['word_min_len'],
                                                    best_params['remove_digits'],
                                                    best_params['use_stemmer_lemmmatizer'],
                                                    best_params['ngram_min'],
                                                    best_params['ngram_max'],
                                                    best_params['min_df'],
                                                    best_params['use_idf'],
                                                    best_params['tfidf_norm'],
                                                    sparse_data)
                                                   
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
        
        iteration_num += 1
        log_computing_progress(time_start, iteration_num, iterations_count, percentage)
        log_message('{0}.outer train-test split, SCORE: {1:.3f}'.format(outer_split_number, score), log_console=False)
        log_message('{0}.outer train-test split, BEST MODEL HYPERPARAMETERS: {1}'.format(outer_split_number, best_params), log_console=False)
        outer_split_number += 1
        if outer_split_number == outer_cv_executes_count:                                                                
            break
                
    log_message('MEAN SCORE: {:.3f}'.format(np.mean(outer_scores)))
    for idx, score, best_params in zip(range(outer_cv.get_n_splits()), outer_scores, outer_best_params):
        log_message('{0}.outer train-test split, SCORE: {1:.3f}'.format(idx, score))
        log_message('{0}.outer train-test split, BEST MODEL HYPERPARAMETERS: {1}'.format(idx, best_params))
    log_message('Finish cross_validation_and_grid_search, time duration: {}'.format(timedelta2str(datetime.now() - time_start)))
                
    return np.array(outer_scores), outer_best_params

# for extract most frequancy class labels and bug feature values
def get_most_common(counter, min_count=2):
    most_common_list = []
    most_common = counter.most_common()
    for value, count in most_common:
        if count >= min_count:
            most_common_list.append(value)
    return most_common_list

# main function - process bugs data and train classifiers
def train_classifiers():  
    global USE_NB, USE_LR, USE_SVC, USE_RF, USE_KNN
    global MAX_WORDS
    global MOST_COMMON_CLASSES_COUNT_LIST
    global MIN_TIMES_COUNT_IN_FEATURE         
    global DF_BASE    
    global INNER_CV_SPLITS_COUNT, INNER_CV_EXECUTES_COUNT
    global OUTER_CV_SPLITS_COUNT, OUTER_CV_EXECUTES_COUNT
    global USE_PLATFORM, USE_OP_SYS, USE_REPORTER, USE_DESCRIPTION
    global NOT, STEMMER, LEMMATIZER
    global PRODUCT, COMPONENT, SUMMARY, PLATFORM, OP_SYS, REPORTER, DESCRIPTION
    global CLASS_LABEL
    global OTHER_CLASSES, OTHER_PLATFORMS, OTHER_OS, OTHER_REPORTERS
                    
    for MOST_COMMON_CLASSES_COUNT in MOST_COMMON_CLASSES_COUNT_LIST:                                    
        log_message('\nMost classes count: {}, time: {}'.format(MOST_COMMON_CLASSES_COUNT, 
              datetime.now().strftime('%H:%M:%S')))      
                
        df = DF_BASE
        
        # process bugs data
        
        target_class = df[CLASS_LABEL].value_counts()        
        target_class_counter = collections.Counter(target_class.to_dict())
        target_class_most_common = target_class_counter.most_common(MOST_COMMON_CLASSES_COUNT)                
        target_class_labels_most_common = []
        for class_label in target_class_most_common:
            target_class_labels_most_common.append(class_label[0])
        if CREATE_VIRTUAL_CLASS_OTHER == True:
            df.loc[~df[CLASS_LABEL].isin(target_class_labels_most_common), CLASS_LABEL] = OTHER_CLASSES
        else:                       
            df.drop(df.index[~df[CLASS_LABEL].isin(target_class_labels_most_common)], inplace=True)            
        target_class = df[CLASS_LABEL].value_counts()
        target_class_counter = collections.Counter(target_class.to_dict())
        target_class_most_common = target_class_counter.most_common()                                               
        target_class2index = collections.defaultdict(int)                       
        for cid, class_label in enumerate(target_class_most_common):
            target_class2index[class_label[0]] = cid  
        #index2target_class = {v:k for k, v in target_class2index.items()}    
        classes_count = len(target_class2index)
        log_message('Classes count: {}'.format(classes_count))        
        
        platform = df[PLATFORM].value_counts()
        platform_counter = collections.Counter(platform.to_dict())            
        platform_most_common = get_most_common(platform_counter,
                                               min_count=MIN_TIMES_COUNT_IN_FEATURE)        
        df.loc[~df[PLATFORM].isin(platform_most_common), PLATFORM] = OTHER_PLATFORMS
                        
        opsys = df[OP_SYS].value_counts()
        opsys_counter = collections.Counter(opsys.to_dict())                      
        opsys_most_common = get_most_common(opsys_counter,
                                            min_count=MIN_TIMES_COUNT_IN_FEATURE)
        df.loc[~df[OP_SYS].isin(opsys_most_common), OP_SYS] = OTHER_OS
        
        reporter = df[REPORTER].value_counts()
        reporter_counter = collections.Counter(reporter.to_dict())          
        reporter_most_common = get_most_common(reporter_counter,
                                               min_count=MIN_TIMES_COUNT_IN_FEATURE)   
        df.loc[~df[REPORTER].isin(reporter_most_common), REPORTER] = OTHER_REPORTERS
                                           
        featureCount = 1#SUMMARY
        if USE_PLATFORM == True:
            featureCount += 1
        if USE_OP_SYS == True:
            featureCount += 1
        if USE_REPORTER == True:
            featureCount += 1        
            
        x = np.empty((df.shape[0], featureCount), dtype=object) 
        y = np.empty((df.shape[0],), dtype=int)
        
        row_num = 0         
        for index, row in df.iterrows():
            column_num = 0
            full_description = str(row[SUMMARY])        
            if USE_DESCRIPTION == True:
                full_description = full_description + ' ' + str(row[DESCRIPTION])
            full_description = clean_lower_string(full_description)
            x[row_num][column_num] = full_description
            column_num += 1
                
            if USE_PLATFORM == True:
                x[row_num][column_num] = str(row[PLATFORM])
                column_num += 1
            
            if USE_OP_SYS == True:
                x[row_num][column_num] = str(row[OP_SYS])            
                column_num += 1
            
            if USE_REPORTER == True:
                x[row_num][column_num] = str(row[REPORTER])
                column_num += 1
            
            label = row[CLASS_LABEL]
            idx = target_class2index[label]
            y[row_num] = idx
            row_num += 1
                
        del df
        
        # train classifiers
        
        # Naive Bayes
        if TRAIN_NB:
            log_message('---NB---')
            inner_cv = StratifiedKFold(INNER_CV_SPLITS_COUNT, shuffle=True,
                                       random_state=RANDOM_STATE)
            outer_cv = StratifiedKFold(INNER_CV_SPLITS_COUNT, shuffle=True, 
                                       random_state=RANDOM_STATE)    
            classifier = MultinomialNB                  
            param_grid = {'max_words': [MAX_WORDS],
                          'word_min_len': [1],
                          'remove_digits': [False],
                          'use_stemmer_lemmmatizer': [LEMMATIZER],#NOT, STEMMER],
                          'ngram_min': [1],
                          'ngram_max': [1],
                          'min_df': [5],
                          'use_idf': [True],
                          'tfidf_norm': ['l2'],
                          #--- Classifier Params ---
                          'alpha' : [0.05],
                          'fit_prior': [True],                          
                          'class_prior' : [None]
                          } 
            nb_scores, nb_best_params = cross_validation_and_grid_search(x, y,  
                      classifier, inner_cv, INNER_CV_EXECUTES_COUNT,
                      outer_cv, OUTER_CV_EXECUTES_COUNT,
                      ParameterGrid(param_grid), classes_count=classes_count)                
                                     
        # Logistic Regression 
        if TRAIN_LR == True:
            log_message('---LR---')
            inner_cv = StratifiedKFold(INNER_CV_SPLITS_COUNT, shuffle=True,
                                       random_state=RANDOM_STATE)
            outer_cv = StratifiedKFold(INNER_CV_SPLITS_COUNT, shuffle=True,
                                       random_state=RANDOM_STATE)    
            classifier = LogisticRegression                       
            param_grid = {'max_words': [MAX_WORDS],
                          'word_min_len': [1],
                          'remove_digits': [False],
                          'use_stemmer_lemmmatizer': [LEMMATIZER],#NOT, STEMMER],
                          'ngram_min': [1],
                          'ngram_max': [1],
                          'min_df': [5],
                          'use_idf': [True],
                          'tfidf_norm': ['l2'],
                          #--- Classifier Params ---
                          'penalty' : ['l2'],
                          'C': [10],
                          'multi_class' : ['multinomial'],
                          'solver' : ['saga'],
                          'max_iter': [500],
                          'n_jobs' : [-1],
                          'random_state' : [RANDOM_STATE]
                          } 
            lr_scores, lr_best_params = cross_validation_and_grid_search(x, y, 
                      classifier, inner_cv, INNER_CV_EXECUTES_COUNT,
                      outer_cv, OUTER_CV_EXECUTES_COUNT,
                      ParameterGrid(param_grid), classes_count=classes_count)
                                                
        # Support Vector Machine with Core
        if TRAIN_SVC:
            log_message('---SVC---')
            inner_cv = StratifiedKFold(INNER_CV_SPLITS_COUNT, shuffle=True,
                                       random_state=RANDOM_STATE)
            outer_cv = StratifiedKFold(INNER_CV_SPLITS_COUNT, shuffle=True,
                                       random_state=RANDOM_STATE)
            classifier = SVC
            param_grid = {'max_words': [MAX_WORDS],
                          'word_min_len': [1],
                          'remove_digits': [False],
                          'use_stemmer_lemmmatizer': [LEMMATIZER],#NOT, STEMMER],
                          'ngram_min': [1],
                          'ngram_max': [5],
                          'min_df': [5],
                          'use_idf': [True],
                          'tfidf_norm': ['l2'],
                          #--- Classifier Params ---
                          'kernel': ['rbf'],
                          'C': [10],
                          'gamma': [0.05],
                          'random_state' : [RANDOM_STATE]
                         } 
            svc_scores, svc_best_params = cross_validation_and_grid_search(x, y, 
                      classifier, inner_cv, INNER_CV_EXECUTES_COUNT,
                      outer_cv, OUTER_CV_EXECUTES_COUNT,
                      ParameterGrid(param_grid), classes_count=classes_count)
        
        # Random Forest
        if TRAIN_RF == True:
            log_message('---RF---')
            inner_cv = StratifiedKFold(INNER_CV_SPLITS_COUNT, shuffle=True,
                                       random_state=RANDOM_STATE)
            outer_cv = StratifiedKFold(INNER_CV_SPLITS_COUNT, shuffle=True,
                                       random_state=RANDOM_STATE)
            classifier = RandomForestClassifier
            param_grid = {'max_words': [MAX_WORDS],
                          'word_min_len': [1],
                          'remove_digits': [False],
                          'use_stemmer_lemmmatizer': [LEMMATIZER],#NOT, STEMMER],
                          'ngram_min': [1],
                          'ngram_max': [5],
                          'min_df': [5],
                          'use_idf': [True],
                          'tfidf_norm': ['l2'],
                          #--- Classifier Params ---
                          'n_estimators' : [500],
                          'criterion' : ['gini'],
                          'max_features': [0.1],
                          'max_depth':[None],
                          'oob_score': [False],
                          'min_samples_split': [5],
                          'min_samples_leaf': [1],
                          'class_weight' : [ None],
                          'n_jobs' : [-1],
                          'random_state' : [RANDOM_STATE]
                         }                      
            rf_scores, rf_best_params = cross_validation_and_grid_search(x, y, 
                      classifier, inner_cv, INNER_CV_EXECUTES_COUNT,
                      outer_cv, OUTER_CV_EXECUTES_COUNT,
                      ParameterGrid(param_grid), classes_count=MOST_COMMON_CLASSES_COUNT)

        # K-Nearest-Neighbors
        if TRAIN_KNN:
            log_message('---KNN---')
            inner_cv = StratifiedKFold(INNER_CV_SPLITS_COUNT, shuffle=True,
                                       random_state=RANDOM_STATE)
            outer_cv = StratifiedKFold(INNER_CV_SPLITS_COUNT, shuffle=True,
                                       random_state=RANDOM_STATE)    
            classifier = KNeighborsClassifier                 
            param_grid = {'max_words': [MAX_WORDS],
                          'word_min_len': [1],
                          'remove_digits': [False],
                          'use_stemmer_lemmmatizer': [LEMMATIZER],#NOT, STEMMER],
                          'ngram_min': [1],
                          'ngram_max': [1],
                          'min_df': [5],
                          'use_idf': [True],#False
                          'tfidf_norm': ['l2'],
                          #--- Classifier Params ---
                          'n_neighbors' : [10],                                                   
                          'weights': ['distance'],                         
                          'algorithm' : ['ball_tree'],
                          'leaf_size': [3],
                          'p' : [2],# 2],
                          'n_jobs' : [3]
                          }         
            knn_scores, knn_best_params = cross_validation_and_grid_search(x, y, 
                      classifier, inner_cv, INNER_CV_EXECUTES_COUNT,
                      outer_cv, OUTER_CV_EXECUTES_COUNT,
                      ParameterGrid(param_grid), classes_count=classes_count, 
                      sparse_data=False)


log_message('Start train tfidf-based classifiers, time: {}'.format(datetime.now().strftime('%H:%M:%S'))) 
train_classifiers()
log_message('Finish train tfidf-based classifiers, time: {}'.format(datetime.now().strftime('%H:%M:%S')), close_log_file=True)


