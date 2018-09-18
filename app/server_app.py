import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, request, Response
import operator

SCRIPTS_DIR_C = os.path.abspath(os.path.join(os.getcwd(), 'scripts'))
if os.path.exists(SCRIPTS_DIR_C):
    if SCRIPTS_DIR_C not in sys.path:
        sys.path.append(SCRIPTS_DIR_C)

#------------------------
import configuration as config
import utils
from logger import Logger
from storage import Storage
#-----------------------

#----- Helper Functions


def json_error_msg(error_msg, bug_id=None):
    msg = {config.API_JSON_ERROR_MSG: error_msg}
    if bug_id is not None:
        msg[config.API_BUG_ID] = bug_id
    msg = json.dumps(msg)
    msg = json.loads(msg)
    return msg


def json_bugs_data(bugs_data, not_found_bug_id=None):
    bugs_data[config.STORAGE_COLUMN_CONFIDENCE] = bugs_data[config.STORAGE_COLUMN_CONFIDENCE].round(
        config.API_CONFIDENCE_SIGNIFICANT_DIGITS)
    results = bugs_data.to_json(orient='records')
    results = json.loads(results)
    bugs_data = {config.API_JSON_RESULTS: results}
    if not_found_bug_id is not None:
        bugs_data[config.API_NOT_FOUND_BUG_ID] = not_found_bug_id
    bugs_data = json.dumps(bugs_data)
    bugs_data = json.loads(bugs_data)
    return bugs_data


def bugs_data_as_json(bugs_data, not_found_bugs=None):
    results = []
    for item in bugs_data:
        component, confidence = list(
            sorted(item[config.STORAGE_COLUMN_PREDICTIONS].items(), key=lambda x: x[1], reverse=True))[0]
        results.append({
            config.API_BUG_ID: item[config.STORAGE_COLUMN_ID],
            config.API_SUMMARY: item[config.STORAGE_COLUMN_SUMMARY],
            config.API_SUGGESTED_COMPONENT: component,
            config.API_CONFIDENCE: '{:.{}f}'.format(float(confidence), config.API_CONFIDENCE_SIGNIFICANT_DIGITS)
        })
    results = {config.API_JSON_RESULTS: results}
    if not_found_bugs is not None and len(not_found_bugs) > 0:
        results[config.API_NOT_FOUND_BUG_ID] = not_found_bugs
    # Sort the results by confidence
    results['results'] = sorted(results['results'],
                                key=operator.itemgetter('confidence'),
                                reverse=True)
    return json.loads(json.dumps(results))



#----- API
app = Flask(__name__)

logger = Logger(log_dir=config.LOG_DIR, log_file_base_name='server_app')

storage = Storage(logger)
# if storage.tryConnect():
#    storage.selectBugsFromTable()


@app.route('/')
def index():
    message = 'bugs classifier'
    logger.info(message)
    return message


def log_response(message, status):
    logger.info(message)
    return Response(message, status=status, mimetype='application/json')


@app.route('/component', methods=['GET'])
def component():
    status_code = 400
    response = None
    try:
        # extract request parameters
        if request.content_type == 'application/json':
            bug_data = request.get_json(silent=True)
        else:
            bug_data = request.form.to_dict()
            if not bug_data:
                bug_data = request.args.to_dict()

        # bug info is empty
        if (not bug_data) or (bug_data is None):
            response = json_error_msg('Invalid request, specify bug_id or start/end dates')

        # bug_id
        elif config.API_BUG_ID in bug_data:
            # check, one bug_id or list of bug_id
            bug_id = bug_data[config.API_BUG_ID].split(',')
            bug_id = utils.to_int_list(bug_id)

            # invalid bug ID(s)
            if bug_id is None:
                response = json_error_msg('Invalid request, bug_id must be integer value')

            # valid bug ID(s)
            else:
                bug_id = set(bug_id)
                bugs_id_count = len(bug_id)
                # one bug_id
                if bugs_id_count == 1:
                    bug_id = list(bug_id)[0]

                    bugs_data = storage.loadPredictionsForListFromTable([bug_id])

                    if bugs_data is None or len(bugs_data) == 0:
                        status_code = 404
                        response = json_error_msg(
                            'There is no bug with ID: {}'.format(bug_id), bug_id)
                    else:
                        status_code = 200
                        response = bugs_data_as_json(bugs_data)

                # bug_id list
                else:

                    bugs_data = storage.loadPredictionsForListFromTable(list(bug_id))

                    if bugs_data is None or len(bugs_data) == 0:
                        status_code = 404
                        response = json_error_msg(
                            'There are no bugs with requested IDs', list(bug_id))

                    else:
                        status_code = 200
                        bugs_count = len(bugs_data)
                        not_found_bug_id = None
                        if bugs_count < bugs_id_count:
                            not_found_bug_id = list(
                                bug_id - set([item[config.STORAGE_COLUMN_ID] for item in bugs_data]))
                        response = bugs_data_as_json(bugs_data, not_found_bug_id)

        # bug start/end dates
        elif (config.API_START_DATE in bug_data) and (config.API_END_DATE in bug_data):
            start_date = utils.to_date(bug_data[config.API_START_DATE], config.DATE_FORMAT)
            end_date = utils.to_date(bug_data[config.API_END_DATE], config.DATE_FORMAT)

            # start_date or(and) end_date are invalid
            if (start_date is None) or (end_date is None):
                response = json_error_msg(
                    'Invalid request, start/end dates should be {}'.format(config.API_DATE_FORMAT_MESSAGE))

            #start_date > end_date
            elif start_date > end_date:
                response = json_error_msg(
                    'Invalid request, start date should be less or equal end date')

            # start_date and end_date are valid
            else:

                bugs_data = storage.loadPredictionsForDateFromTable(start_date, end_date)

                if bugs_data is None or len(bugs_data) == 0:
                    status_code = 404
                    response = json_error_msg(
                        'There are no bugs between dates [{0}; {1}]'.format(start_date, end_date))
                else:
                    status_code = 200
                    response = bugs_data_as_json(bugs_data)

        # no bug_id, no start/end dates
        else:
            response = json_error_msg('Invalid request, specify bug_id or start/end dates')

    # server error
    except Exception as ex:
        status_code = 500
        response = json_error_msg('Internal server error: {}'.format(ex))
        logger.error(str(ex))

    return log_response(json.dumps(response), status=status_code)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
