""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/10/27 14:33
"""

class Runner(object):
    def __init__(self, config):
        self.all_args = config['all_args']


    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    def compute(self):
        pass

    def train(self):
        pass

    def save(self):
        pass

    def restore(self):
        pass

    def log_train(self):
        pass

    def log_env(self):
        pass