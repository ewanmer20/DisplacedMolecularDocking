import logging
from log_utils import LogUtils

LogUtils.log_config('logging_test')
a=1
b=2
logging.info('Test printout')
logging.info('Test printout')
logging.info('Test printout')
logging.info('Test printout')
logging.info('%s %s',str(a),str(b))