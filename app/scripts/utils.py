import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import precision_recall_fscore_support

def to_int(value):
    try:
        return int(str(value))
    except ValueError:
        return None

def to_int_list(values_list):
    int_list = []
    for value in values_list:
        int_value = to_int(value)
        if int_value is None:
            return None
        int_list.append(int_value)
    return int_list

def to_date(value, date_format='%Y-%m-%d'):
    try:
        return datetime.strptime(str(value), date_format).date()
    except ValueError:
        return None

def contain(a, x):
    pos = np.searchsorted(a, x)
    if pos >= len(a):
        return False
    elif a[pos] == x:
        return True
    return False

def clean_string(string):            
    string = re.sub(r'[^A-Za-z0-9()\'\`]', ' ', string)    
    return string.strip() 

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
def progress2str(time_start, iteration_num, iterations_count):
    time_passed = datetime.now() - time_start                
    percentage = 100.0 * iteration_num / iterations_count
    message = 'processed: {:>5.1f}%, time passed: {}, total time est.: {}'.format(percentage,
              timedelta2str(time_passed),timetotal2str(time_passed, iteration_num, iterations_count))
    return message, percentage

# calculate computing progress and log it
def log_progress(logger, time_start, iteration_num, iterations_count,
                   last_log_percentage, delta_percentage=1.0):        
    message, percentage = progress2str(time_start, iteration_num, iterations_count)
    if (percentage >= last_log_percentage + delta_percentage) or (percentage >= 100.0):
        last_log_percentage = percentage
        logger.info(message)
    return last_log_percentage

# for extract most frequancy class labels and bug feature values
def get_most_common(counter, min_count=2):
    most_common_list = []
    most_common = counter.most_common()
    for value, count in most_common:
        if count >= min_count:
            most_common_list.append(value)
    return most_common_list


def create_classification_report(y, test_indices, pred, class_indices, class_labels):
    report = precision_recall_fscore_support(y[test_indices], pred,
                                                  average=None,
                                                  labels=class_indices)
    report_avg = precision_recall_fscore_support(y[test_indices], pred,
                                                  average='weighted',
                                                  labels=class_indices)        
    class_report = pd.DataFrame(data={'precision' : report[0],
                                               'recall' : report[1],
                                               'f1_score' : report[2],
                                               'support' : report[3]},
    index=class_labels).append(pd.DataFrame(data={'precision' : np.mean(report_avg[0]),
                                               'recall' : np.mean(report_avg[1]),
                                               'f1_score' : np.mean(report_avg[2]),
                                               'support' : np.sum(report_avg[3])},index=['weighted_avg / total']))
    class_report.at['weighted_avg / total', 'support'] = class_report['support'].sum()    
    class_report = class_report[['recall', 'precision', 'f1_score', 'support']]     
    return class_report

def format_classification_report(report, sorted_by='recall', ascending=False,
                                 in_percentages=True, digits=1):
    str_report = report.copy()
    if in_percentages == True:
        str_report['precision'] = 100 * str_report['precision']
        str_report['recall'] = 100 * str_report['recall']
        str_report['f1_score'] = 100 * str_report['f1_score']        
    str_report['precision'] = str_report['precision'].round(digits)
    str_report['recall'] = str_report['recall'].round(digits)    
    str_report['f1_score'] = str_report['f1_score'].round(digits)
    total = str_report.loc['weighted_avg / total'].copy()
    str_report.drop('weighted_avg / total', inplace=True)
    str_report.sort_values(by=[sorted_by], ascending=ascending, inplace=True)
    str_report = str_report.append(total)
    str_report['support'] = str_report['support'].astype('int64')           
    return str_report

def classification_report2string(report, sorted_by='recall', ascending=False,
                                 in_percentages=True, digits=1):
    str_report = format_classification_report(report, sorted_by=sorted_by, ascending=ascending,
                                 in_percentages=in_percentages, digits=digits)
    return '{}'.format(str_report)

def classification_report2csv(file_path, report, index_label='Component',
                              sorted_by='recall', ascending=False,
                              in_percentages=True, digits=1):
    csv_report = format_classification_report(report, sorted_by=sorted_by, ascending=ascending,
                                 in_percentages=in_percentages, digits=digits)
    csv_report.to_csv(file_path, index_label=index_label)
