import math
import os
import random
import sys
import time

import numpy as np


try:
    # python2 支持
    reload
except NameError:
    # py3k has unicode by default
    pass
# 在 try 执行成功的情况下才执行 else
else:
    # python2 设置编码
    reload(sys).setdefaultencoding('utf-8')
    
try:
    # python2 支持
    from ConfigParser import SafeConfigParser
except:
    # python3 支持
    from configparser import SafeConfigParser

def get_config(config_file='seq2seq.ini'):
    parser = SafeConfigParser()
    parser.read(config_file)
    # get the ints, floats and strings
    _conf_ints = [ (key, int(value)) for key,value in parser.items('ints') ]
    _conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
    _conf_strings = [ (key, str(value)) for key,value in parser.items('strings') ]
    return dict(_conf_ints + _conf_floats + _conf_strings)