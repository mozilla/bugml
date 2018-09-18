import os
from datetime import datetime

INFO = 'Info'
WARNING = 'Warning'
ERROR = 'Error'
MESSAGE_NUM_WIDTH = 4
MESSAGE_DATETIME_WIDTH = 18
MESSAGE_TYPE_WIDTH = 7


class Logger:
    log_stream = None
    log_message_number = None

    def __init__(self, log_dir, log_file_base_name):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # os.path.isdir(sDir):
        log_file_path = log_dir + '/' + log_file_base_name + '_log_' + \
            datetime.now().strftime('%Y_%m_%d__%H_%M_%S') + '.txt'
        self.log_stream = open(log_file_path, 'w')
        self.log_message_number = 0

    def __del__(self):
        self.log_stream.close()

    # log message to console output and to the log file
    def __log_message(self, message_type, message, log_to_console):
        self.log_message_number += 1
        log_message = '{message: >{width}} | '.format(
            message=self.log_message_number, width=MESSAGE_NUM_WIDTH)
        log_message += '{message: >{width}} | '.format(
            message=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), width=MESSAGE_DATETIME_WIDTH)
        log_message += '{message: >{width}} | '.format(
            message=message_type, width=MESSAGE_TYPE_WIDTH) + message
        self.log_stream.write(log_message + '\n')
        self.log_stream.flush()
        if log_to_console == True:
            print(log_message)

    def info(self, message, log_to_console=True):
        self.__log_message(INFO, message, log_to_console)

    def warning(self, message, log_to_console=True):
        self.__log_message(WARNING, message, log_to_console)

    def error(self, message, log_to_console=True):
        self.__log_message(ERROR, message, log_to_console)
