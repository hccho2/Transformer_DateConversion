import atexit
from datetime import datetime
import json
from threading import Thread
from urllib.request import Request, urlopen
import logging, os

_format = '%Y-%m-%d %H:%M:%S.%f'
_file = None
_run_name = None
_slack_url = None


def init(filename, run_name, slack_url=None):
    global _file, _run_name, _slack_url
    _close_logfile()
    _file = open(filename, 'a')
    _file.write('\n-----------------------------------------------------------------\n')
    _file.write('Starting new training run\n')
    _file.write('-----------------------------------------------------------------\n')
    _run_name = run_name
    _slack_url = slack_url


def log(msg, slack=False):
    print(msg)
    if _file is not None:
        _file.write('[%s]    %s\n' % (datetime.now().strftime(_format)[:-3], msg))
    if slack and _slack_url is not None:
        Thread(target=_send_slack, args=(msg,)).start()


def _close_logfile():
    global _file
    if _file is not None:
        _file.close()
        _file = None


def _send_slack(msg):
    req = Request(_slack_url)
    req.add_header('Content-Type', 'application/json')
    urlopen(req, json.dumps({
        'username': 'tacotron',
        'icon_emoji': ':taco:',
        'text': '*%s*: %s' % (_run_name, msg)
    }).encode())


atexit.register(_close_logfile)



def set_tf_log(load_path):
    # Estimator에서의 log가 infolog로 나가지 않기 때문에 별도로 설정해 줌
    # Estimator의 log도 format을 변경. 시간이 출력되도록...
    logger = logging.getLogger('tensorflow')
    log_path = os.path.join(load_path, 'train-tf.log')   # append mode로 열린다.
    FileHandler = logging.FileHandler(log_path)
    StreamHandler =  logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    StreamHandler.setFormatter(formatter)
    FileHandler.setFormatter(formatter)
    
    
    logger.addHandler(StreamHandler)
    logger.addHandler(FileHandler)