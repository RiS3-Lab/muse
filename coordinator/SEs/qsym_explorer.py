#!/usr/bin/env python
import csv
import sys
import utils
import ConfigParser
from utils import bcolors

from qsym import *

class QsymExplorer:
    def __init__(self, config, proj_dir):
        self.se_factory = list()
        self.config = config
        self.proj_dir = proj_dir
        self.get_qsym_configs()

    def get_qsym_configs(self):
        config = ConfigParser.ConfigParser()
        config.read(self.config)
        self.se_factory.append(Qsym(self.config, self.proj_dir))

        #QSYM is not using this coverage file
        self.fuzzer_cov_file = config.get("auxiliary info", "cov_edge_file").replace("@target",self.proj_dir)
        utils.rmfile_force(self.fuzzer_cov_file)

    def get_se_size(self):
        return len(self.se_factory)

    def get_heuristics(self):
        #no search heuristics for qsym
        return []

    def get_fuzzer_cov_file(self):
        return self.fuzzer_cov_file


    def is_explorers_alive(self):
        alive = False
        for _ in self.se_factory:
            alive = alive or _.alive()
        return alive


    def run(self, input_list, cov_file_list, batch_run_input_num):
        records = []
        for _ in self.se_factory:
            r = _.run(input_list, cov_file_list)
            records.append(r)
            break
        return records

    def stop(self):
        for _ in self.se_factory:
            _.stop()


    def terminate_callback(self):
        """called when SIGINT and SIGTERM"""
        for _ in self.se_factory:
            _.terminate_callback()

    def periodic_callback(self):
        """called every 1 hour"""
        for _ in self.se_factory:
            _.periodic_callback()
