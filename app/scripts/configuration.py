import os
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

DATA_DIR_C = os.path.abspath(os.path.join(os.getcwd(), 'data'))
MODELS_DIR_C = os.path.abspath(os.path.join(os.getcwd(), 'models'))
LOGS_DIR_C = os.path.abspath(os.path.join(os.getcwd(), 'logs'))
CONFIG_DIR_C = os.path.abspath(os.path.join(os.getcwd(), 'config'))
SCRIPTS_DIR_C = os.path.abspath(os.path.join(os.getcwd(), 'scripts'))

DATA_FILENAME_C = 'bugDataTest500.csv'

UNTRIAGED_FILE_C = 'untriagedBugsData.csv'

CONFIG_FILENAME_C = 'server_config.ini'


# path to csv file with bugs features (desription, reporter, etc)
INPUT_DATA_FILE_PATH = os.path.join(DATA_DIR_C, DATA_FILENAME_C)
MODEL_DIR = MODELS_DIR_C  # os.path.dirname(os.path.abspath( __file__ )) + '/models'
LOG_DIR = LOGS_DIR_C  # os.path.dirname(os.path.abspath( __file__ )) + '/logs'


PRODUCT = 'product'
COMPONENT = 'component'
SUMMARY = 'short_desc'
PLATFORM = 'rep_platform'
OP_SYS = 'op_sys'
REPORTER = 'reporter'
DESCRIPTION = 'description'
BUGS_DATA_SEPARATOR = ','

# text preprocessing options
NOT = 'not'
STEMMER = 'stemmer'
LEMMATIZER = 'lemmatizer'

# real feature values   rename to these virtual feature values
OTHER_PLATFORMS = 'OtherPlatforms'
OTHER_OP_SYS = 'OtherOpSys'
OTHER_REPORTERS = 'OtherReporters'

# count of splits in cross-validation and grid search models hyperparams

TEST_SIZE = 0.10
VALID_SIZE = 0.10

GRID_SEARCH_SPLITS_COUNT = 5
GRID_SEARCH_EXECUTES_COUNT = 5

CROSS_VALIDATION_SPLITS_COUNT = 10
CROSS_VALIDATION_EXECUTES_COUNT = 10

RANDOM_STATE = 42

READ_DATA_ROWS_MAX_COUNT = 500000
COMPONENTS_MAX_COUNT = 1000
VOCABULARY_WORDS_MAX_COUNT = 50000
# the min repeat count of feature (reporter, platform, OS) value in feature
FEATURE_VALUE_MIN_FREQUENCY = 5
COMPONENT_MIN_BUGS_COUNT = 50

FEATURES_COUNT = 4
DESCRIPTION_COLUMN_NUM = 0
PLATFORM_COLUMN_NUM = 1
OP_SYS_COLUMN_NUM = 2
REPORTER_COLUMN_NUM = 3

TEXT_PROCESSING_PARAMETERS = {'max_words': VOCABULARY_WORDS_MAX_COUNT,
                              'word_min_len': 1,
                              'remove_digits': False,
                              'use_stemmer_lemmmatizer': STEMMER,
                              'ngram_min': 1,
                              'ngram_max': 5,
                              'min_df': 5,
                              'use_idf': True,
                              'tfidf_norm': 'l2'}

CLASSIFIER_NAMES_DICT = {SGDClassifier: 'SGD',
                         SVC: 'SVC'}

SGD_PARAMETERS = {'loss': 'modified_huber',
                  'alpha': 0.0002,
                  'penalty': 'l2',
                  'tol': 1e-5,
                  'max_iter': 1000,
                  'n_jobs': -1,
                  'random_state': RANDOM_STATE}

SGD_PARAMETERS_GRID = {'max_words': [VOCABULARY_WORDS_MAX_COUNT],
                       'word_min_len': [1],
                       'remove_digits': [False],
                       'use_stemmer_lemmmatizer': [STEMMER],
                       'ngram_min': [1],
                       'ngram_max': [5],
                       'min_df': [5],
                       'use_idf': [True],
                       'tfidf_norm': ['l2'],
                       # classifier parameters
                       'loss': ['modified_huber'],
                       'alpha': [0.0002],
                       'penalty': ['l2', 'elasticnet'],
                       'tol': [1e-5],
                       'max_iter': [1000],
                       'n_jobs': [-1],
                       'random_state': [RANDOM_STATE]}

SVC_PARAMETERS = {'kernel': 'rbf',
                  'C': 10,
                  'gamma': 0.05,
                  'probability': True,
                  'cache_size': 256,
                  'max_iter': 1000,
                  'random_state': RANDOM_STATE}


ParametersRanges = {'batch_size': [32],
                    'dropout': [0.15],
                    'conv_size': [3, 5],
                    'epochs': [10],
                    'optimizer': ['nadam'],
                    'conv_filters': [256],
                    'pooling': ['local'],
                    'padding': ['same'],
                    'activation': ['relu', 'LeakyRelu(0.2)'],
                    'lstm_units': [80],
                    # use pretrained glove vector (1) or train from zero (0).
                    'use_pretrained': [0],
                    # 1 for single model, > 1 for ensemble of models, trained on parts of dataset.
                    'split_count': [1],
                    # just a name suffix. May be used to train same models few times.
                    'name_suffix': ['suffixCnn2']
                    }

cfgModelsSection = 'Models'
cfgDownloadSection = 'Download'

cfgElementProducts = 'UntriagedProducts'
cfgElementInterval = 'Interval'
cfgElementState = 'State'

cfgValueStateInactive = 'Inactive'
cfgValueStateBusy = 'Busy'
cfgValueStateStop = 'Stop'

DATASET_SUFFIX = 'BugData'

DB_HOST_C = 'storage_container'
DB_PORT_C = 33060
DB_USER_C = 'root'
DB_PASS_C = 'jwoe_521Mdb'

DB_NAME_C = 'BugsDB'
BUGS_TABLE_NAME_C = 'BugsDataTable'

# dict with tables, which will be used in configuration process of db.
#TABLES = {}

# useful info: https://www.bugzilla.org/docs/2.16/html/dbschema.html

# --- Storage settings
STORAGE_HOST = DB_HOST_C  # 'localhost'
STORAGE_PORT = DB_PORT_C  # 33060
STORAGE_USER = DB_USER_C  # 'svm'
STORAGE_PASSWORD = DB_PASS_C  # '1234'
STORAGE_CHARSET = 'utf8mb4'
STORAGE_CHARSET_COLLATION = 'utf8mb4_unicode_ci'
STORAGE_DATABASE_NAME = DB_NAME_C  # 'BugsDB'
STORAGE_BUGS_TABLE_NAME = BUGS_TABLE_NAME_C  # 'BugsTable'

STORAGE_CONNECT_INTERVAL = 15
STORAGE_CONNECT_RETRY_COUNT = 5


#DB_HOST_C = STORAGE_HOST
#DB_PORT_C = STORAGE_PORT
#DB_USER_C = STORAGE_USER
#DB_PASS_C = STORAGE_PASSWORD
#DB_NAME_C = STORAGE_DATABASE_NAM
#TABLE_NAME_C = STORAGE_BUGS_TABLE_NAME

STORAGE_COLUMN_ID = 'bug_id'
STORAGE_COLUMN_PRODUCT = PRODUCT
STORAGE_COLUMN_COMPONENT = COMPONENT
STORAGE_COLUMN_SUMMARY = SUMMARY
STORAGE_COLUMN_PLATFORM = PLATFORM
STORAGE_COLUMN_OP_SYS = OP_SYS
STORAGE_COLUMN_REPORTER = REPORTER
STORAGE_COLUMN_DESCRIPTION = DESCRIPTION
STORAGE_COLUMN_OPENDATE = 'opendate'
STORAGE_COLUMN_PREDICTIONS = 'predictions'

STORAGE_COLUMN_DATE = 'date'
STORAGE_COLUMN_SUGGESTED_COMPONENT = 'suggested_component'
STORAGE_COLUMN_CONFIDENCE = 'confidence'

BUGS_TABLE_COLUMNS_C = {STORAGE_COLUMN_ID: 'mediumint(9) NOT NULL',
                        STORAGE_COLUMN_OPENDATE: 'datetime',
                        #                  "keywords" : 'mediumtext',
                        #                  "priority" : 'varchar(32)',
                        STORAGE_COLUMN_PRODUCT: 'varchar(64)',
                        STORAGE_COLUMN_COMPONENT: 'varchar(50)',
                        #                  "bug_status" : 'varchar(64)',
                        #                  "resolution" : 'varchar(64)',
                        #                  "version" : 'varchar(16)',
                        STORAGE_COLUMN_SUMMARY: 'mediumtext NOT NULL',
                        STORAGE_COLUMN_PLATFORM: 'varchar(64)',
                        STORAGE_COLUMN_OP_SYS: 'varchar(64)',
                        STORAGE_COLUMN_REPORTER: 'varchar(255) NOT NULL',
                        STORAGE_COLUMN_DESCRIPTION: 'mediumtext',
                        STORAGE_COLUMN_PREDICTIONS: 'varchar(255)'
                        }


# --- API settings
API_BUG_ID = STORAGE_COLUMN_ID
API_SUGGESTED_COMPONENT = 'suggested_component'
API_CONFIDENCE = 'confidence'
API_SUMMARY = SUMMARY


API_NOT_FOUND_BUG_ID = 'not_found_bug_id'
API_START_DATE = 'start_date'
API_END_DATE = 'end_date'
DATE_FORMAT = '%Y-%m-%d'
API_DATE_FORMAT_MESSAGE = 'YYYY-MM-DD'
API_CONFIDENCE_SIGNIFICANT_DIGITS = 3
API_JSON_RESULTS = 'results'
API_JSON_ERROR_MSG = 'error_msg'
