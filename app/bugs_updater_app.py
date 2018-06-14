import os
import sys

SCRIPTS_DIR_C = os.path.abspath(os.path.join(os.getcwd(), 'scripts'))
if os.path.exists(SCRIPTS_DIR_C):
    if SCRIPTS_DIR_C not in sys.path:
        sys.path.append(SCRIPTS_DIR_C)
else:
    raise AssertionError('Cannot find scripts directory: ' + SCRIPTS_DIR_C)

import configuration as config
from BugsDataUpdater import runBugsDataUpdater

    
def initApp():
    if not os.path.exists(config.LOGS_DIR_C):
        os.makedirs(config.LOGS_DIR_C)
    if not os.path.exists(config.CONFIG_DIR_C):
        os.makedirs(config.CONFIG_DIR_C)
    if not os.path.exists(config.DATA_DIR_C):
        os.makedirs(config.DATA_DIR_C)
    if not os.path.exists(config.MODELS_DIR_C):
        os.makedirs(config.MODELS_DIR_C)

    if os.path.exists(config.MODELS_DIR_C):
        if config.MODELS_DIR_C not in sys.path:
            sys.path.append(config.MODELS_DIR_C)
    
        
initApp()
runBugsDataUpdater()
