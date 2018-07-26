import gc
from sklearn.linear_model import SGDClassifier

from logger import Logger
import utils
import configuration as config
from bugclassifier import BugClassifier


# run once now with undefined set_params
# def set_params(params):

# Returns float score between 0 and 1.
def test_model(modelName, datasetName, px_test, py_test, sequenceInput=None):
    logger = Logger(log_dir=config.LOG_DIR, log_file_base_name='test_model_' + modelName)

    sProduct = datasetName[:-len(config.DATASET_SUFFIX)]

    bug_classifier = BugClassifier(logger=logger, classifier=SGDClassifier,
                                   classifier_parameters=config.SGD_PARAMETERS,
                                   text_processing_parameters=config.TEXT_PROCESSING_PARAMETERS,
                                   target_product=sProduct,
                                   components_exclude_list=['Untriaged'])

    score, class_report = bug_classifier.load_data_train_test_model(data_file_path=config.INPUT_DATA_FILE_PATH,
                                                                    test_size=config.TEST_SIZE)

    bug_classifier.save_model(config.MODEL_DIR)

    del bug_classifier
    gc.collect()
    return score


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
#    Returns:
# list with dicts with next structure:
# [{
#  'bug_id': 0,
#  'Firefox - Installer': 0.7,
#  'Firefox - General': 0.15,
#  'Firefox - Sync': 0.1,
# }] # dict contains bug_id with int value and string components with float or int probabilities.
def predict_components(testData, modelName, pKeys, pData):
    logger = Logger(log_dir=config.LOG_DIR, log_file_base_name='predict_components_' + modelName)
    try:
        dataColumns = [config.REPORTER, config.SUMMARY,
                       config.OP_SYS, config.PLATFORM, config.DESCRIPTION]
        sProduct = 'Firefox'
        for item in modelName.split('_'):
            k = item.find(config.DATASET_SUFFIX)
            if k > 0:
                sProduct = item[:k]
                break
        model = BugClassifier(logger=logger, classifier=SGDClassifier, target_product=sProduct)
        model.load_model(model_dir=config.MODEL_DIR, classifier=SGDClassifier)
        cat_labels = []
        for item in testData:
            data = {key: value for key, value in zip(dataColumns, item['data'].split(' ||| '))}
            component, probability = model.predict_bug(data)
            componentKey = model.target_product + ' - ' + component
            cat_labels.append({config.STORAGE_COLUMN_ID: item['id'], componentKey: probability})
        del model
        gc.collect()
        return cat_labels
    except Exception as exc:
        logger.error(str(exc))
        return None
