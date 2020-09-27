#!/usr/bin/env python
import math
import utils
import sys
import os
import csv
import random
import ConfigParser
from utils import bcolors
from operator import itemgetter
import itertools
import functools
import numpy
from psutil import virtual_memory
from online_learning import *
from ml_engine import *
from ml_ensemble import *

# Reachable Bug
RB_NORM=10000000.0
# Reachable cov
RC_NORM=100000000.0
# Path length
P_NORM=1000.0
# Undiscovered neighbours
UN_NORM=1000.0
# Input Size
S_NORM=1000.0
# Label
L_NORM=10.0
# Current queue size
Q_NORM=1000.0
# Number of cmps on the path
CN_NORM=10000.0
# Number of external calls on the path
ECN_NORM=10000.0
# Number of instrumentation labels on the path
SLN_NORM=10000.0
# Number of indirect calls on the path
ICN_NORM=100.0
# Number of memops defined by ASAN
MOP_NORM=10000.0
# Coverage difference brought by the input compared 
# with last global coverage
CED_NORM=100.0

def oracle_info(s):
    print bcolors.HEADER+"[Edge-Oracle-Info]"+bcolors.ENDC, "{0}".format(s)

class MeuzzOracle:
    """ This oracle calculate per-seed potential to reach bugs"""
    def __init__(self, config, target_bin):
        #map[edge]={'bbl','dom_num'}, singleton, constantly updating
        self.edge_dom_map = dict()
        #map['BBID']={'DOMNUM'}, singleton, load once
        self.bb_bug_map = None
        #map['edge id']={}
        self.edge_pair_map = dict()
        #map['edge_id] = {'parent_basicblock_id'}
        self.edge_to_parentBB = {}

        self.record_factory = RecordFactory()

        self.config = config
        self.target_prog_cmd = target_bin
        self.target_dir = os.path.dirname(os.path.abspath(self.target_prog_cmd).split()[0])

        self.get_oracle_config()
        self.feature_columns = ['reachable label', 'reachable blocks','path length',
		    'undiscovered neighbours', 'new cov', 'size', 'cmp',
		    'indcall', 'extcall', 'reached labels', 'queue size',
		    'mem ops', 'edge difference']
        self.data_store = DataStore(self.data_store_file, self.feature_columns)
        self.load_bb2bug_file()     #static collection
        self.load_bb2cov_file()     #static collection
        self.load_pair_edges_file() #static collection

        self.fuzzer_input_dir = self.get_fuzzer_queue_dir(config, target_bin)
        oracle_info("Using fuzzer input dir %s" % self.fuzzer_input_dir)
        self.input_to_edges_cache = {} #cache of seed to edges
        self.input_to_reachable_bug_cache = {} #cache of seed to bug score
        self.input_to_reachable_cov_cache = {} #cache of seed to cov score
        self.input_to_feature_cache = {} #{seed : [features]}
        self.covered_fuzzer_edges = set() #edges obtained by replaying savior-afl binary
        self.explored_fuzzer_edges = set() #edges of seeds expored by concolic engine
        self.input_to_num_cmp_cache = {}
        self.input_to_num_indcall_cache = {}
        self.input_to_num_extcall_cache = {}
        self.input_to_num_label_cache = {}
        self.input_to_num_memops_cache = {}

        self.initialize_meuzz()
        self.queue_size = None

        self.initialize_log_fs()

    def initialize_meuzz(self):
        if self.meuzz_variant == 'random_forest':
            oracle_info("Initializing RandomForest model")
            self.meuzz = MLEngine(model_dir=self.save_model, dataset_path=self.init_dataset)
        elif self.meuzz_variant == 'online_learning':
            oracle_info("Initializing OnlineLearning model")
            self.meuzz = OnlineLearningModule(self.save_model, dataset_path=self.init_dataset)
        elif self.meuzz_variant == 'ensemble':
            oracle_info("Initializing EnsembleLearning model")
            self.meuzz = EnsembleLearning(self.save_model, dataset_path=self.init_dataset)
        else:
            utils.error_msg("unknown meuzz variant: {0}".format(self.meuzz_variant))

    def __repr__(self):
        return "meuzz"

    def initialize_log_fs(self):
        if self.use_ramfs:
            # check if we have enough ram to create a 512MB ramdisk
            available = virtual_memory().total
            if available > 2 * 1024 ** 3:
                self.log_base = os.path.join("/tmp", "museram"+str(random.randint(1,100000000)))
                try:
                    utils.mount_ramdisk(self.log_base, 512 * 1024 ** 2)
                    oracle_info("using ramfs [%s] for muse logging" % self.log_base)
                    return
                except Exception:
                    pass

        # fallback to use regular disk in root partition
        oracle_info("using /tmp as base directory for muse logging")
        self.log_base = '/tmp'
        self.use_ramfs = False


    def normalize_label(self, label):
        # Label normalization should be related to the batch run input number
        L_NORM = utils.log2(self.batch_run_input_num * self.batch_run_input_num) + 1
        return int(label)/L_NORM

    def get_fuzzer_queue_dir(self, raw_config, target_bin):
        config = ConfigParser.ConfigParser()
        config.read(raw_config)
        target_dir = os.path.dirname(os.path.abspath(target_bin).split()[0])
        self.input_mode = 'symfile' if '@@' in target_bin else 'stdin'
        sync_dir = config.get("moriarty", "sync_dir").replace("@target", target_dir)

        # TODO: read seed from all queues
        # prefer to use slave queue
        fuzzer_dir = os.path.join(sync_dir, "slave_000001", "queue")

        return fuzzer_dir

    def read_queue(self):
        return [f for f in os.listdir(self.fuzzer_input_dir) if os.path.isfile(os.path.join(self.fuzzer_input_dir, f))]

    def get_oracle_config(self):
        config = ConfigParser.ConfigParser()
        config.read(self.config)
        self.replay_prog_cmd = config.get("auxiliary info", "replay_prog_cmd").replace("@target",self.target_dir)
        try:
            self.bb_to_bug_file = config.get("auxiliary info", "bbl_bug_map").replace("@target", self.target_dir)
            self.bb_to_cov_file = config.get("auxiliary info", "bbl_cov_map").replace("@target", self.target_dir)
            self.pair_edge_file = config.get("auxiliary info", "pair_edge_file").replace("@target", self.target_dir)
        except Exception:
            utils.error_msg("bbl_cov_map|bbl_bug_map|pair_edge files not found in %s"%self.target_dir)
            # sys.exit(-1)
        try:
            self.count_all_edges = False if config.get("edge oracle", "only_count_covered_edge") == "True" else True
        except Exception:
            self.count_all_edges = False
        try:
            self.data_store_file = config.get("auxiliary info", "data_store_file").replace("@target", self.target_dir)
            oracle_info("Save meuzz data to {0}".format(self.data_store_file))
        except Exception:
            self.data_store_file = None

        try:
            self.batch_run_input_num = int(config.get("moriarty", "batch_run_input_num"))
        except Exception:
            self.batch_run_input_num = 1

        # NOTE: meuzz_window will dynamically adjust to the size of the work queue
        self.meuzz_window = 1

        try:
            self.save_model = config.get("edge oracle", "meuzz_model_file")\
                                    .replace("@target", self.target_dir)
            oracle_info("Persist meuzz model from/to {0}".format(self.save_model))
        except Exception:
            self.save_model = None

        try:
            self.init_dataset = config.get("edge oracle", "meuzz_init_data")\
                                .replace("@target", self.target_dir)
            oracle_info("initalize model with dataset {0}".format(self.init_dataset))
        except Exception:
            self.init_dataset = None


        try:
            self.meuzz_variant = config.get("edge oracle", "meuzz_variant")
        except Exception:
            # default to online learning
            self.meuzz_variant = "online_learning"

        try:
            self.use_ramfs = True if config.get("edge oracle", "meuzz_use_ramfs").lower() == 'true' else False
        except Exception:
            # default using /tmp folder
            self.use_ramfs = False

    def terminate_callback(self):
        oracle_info("Meuzz Oracle terminating")
        self.meuzz.save_model()
        if self.use_ramfs:
            try:
                utils.unmount_ramdisk(self.log_base)
            except Exception:
                utils.error_msg("failed to unmount ramdisk %s" % self.log_base)


    def load_bb2bug_file(self):
        try:
            self.bb_bug_map = dict()
            with open(self.bb_to_bug_file) as b2d_file:
                reader = csv.DictReader(b2d_file, delimiter=',')
                for row in reader:
                    if self.bb_bug_map.has_key(row['BBID']):
                        if self.bb_bug_map[row['BBID']] < row['DOMNUM']:
                            #take the higher one, as dma might have collision
                            self.bb_bug_map[row['BBID']] = row['DOMNUM']
                    else:
                        self.bb_bug_map[row['BBID']] = row['DOMNUM']
            oracle_info('Loading BBL to bug Map %s'%self.bb_to_bug_file)
        except Exception:
            utils.error_msg("can't load bb_bug_map: %s"%self.bb_to_bug_file)
            # sys.exit(-1)

    def load_bb2cov_file(self):
        try:
            self.bb_cov_map = dict()
            with open(self.bb_to_cov_file) as b2d_file:
                reader = csv.DictReader(b2d_file, delimiter=',')
                for row in reader:
                    if self.bb_cov_map.has_key(row['BBID']):
                        if self.bb_cov_map[row['BBID']] < row['DOMNUM']:
                            #take the higher one, as dma might have collision
                            self.bb_cov_map[row['BBID']] = row['DOMNUM']
                    else:
                        self.bb_cov_map[row['BBID']] = row['DOMNUM']
            oracle_info('Loading BBL to cov Map %s'%self.bb_to_cov_file)
        except Exception:
            utils.error_msg("can't load bb_cov_map: %s"%self.bb_to_cov_file)
            # sys.exit(-1)

    def get_pair_if_any(self, e):
        ret = []
        self_group = set([e])
        if self.edge_pair_map.has_key(e):
            my_group = self.edge_pair_map[e]
            ret = list(my_group - self_group)
        return ret

    def load_pair_edges_file(self):
        try:
            with open(self.pair_edge_file, "r") as f:
                for line in f:
                    parent_bbid = line.split(':')[0]
                    edges = line.split(':')[1].split()
                    if len(edges) < 2:
                        continue
                    for i in range(len(edges)):
                        #there could be hash collision, causing duplicated edge ids in different code blocks
                        self.edge_pair_map[edges[i]] = set(edges)
                        self.edge_to_parentBB[edges[i]] = parent_bbid

        except:
            utils.error_msg("can't load pair_edge_file: %s"%self.pair_edge_file)
            sys.exit(-1)

    def get_scores(self, dummy_all_edges, inputs):
        stats = []
        features = self.get_features(inputs)
        for entry in features:
            # calculate potential of each seed
            seed, feature = entry
            stat = {}
            stat['score'] = self.meuzz.predict([feature])
            stat['first_seen'] = seed
            stat['interesting_edges'] = []
            stat['size'] = os.path.getsize(seed)
            stats.append(stat)
        return stats

    def get_reachable_bug(self, seed, _):
        if not self.input_to_reachable_bug_cache.has_key(seed):
            if not self.input_to_edges_cache.has_key(seed):
                self.batch_collect_features(seed, _)
            edges = self.input_to_edges_cache[seed]
            reachable = 0
            for e in set(edges):
                neighbours = self.get_pair_if_any(e)
                for ne in neighbours:
                    if (self.count_all_edges) or (not ne in self.covered_fuzzer_edges):
                        neighbour_bbl_id = str(((int(self.edge_to_parentBB[ne]) >> 1) ^ int(ne)) & 0xfffff)
                        try:
                            ne_score = float(self.bb_bug_map[neighbour_bbl_id])
                            if ne_score > 0:
                                is_interesting_edge = True
                                reachable += ne_score
                        except KeyError:
                            pass
            self.input_to_reachable_bug_cache[seed] = reachable
        return self.input_to_reachable_bug_cache[seed]

    def get_reachable_cov(self, seed, _):
        if not self.input_to_reachable_cov_cache.has_key(seed):
            if not self.input_to_edges_cache.has_key(seed):
                self.batch_collect_features(seed, _)
            edges = self.input_to_edges_cache[seed]
            reachable = 0
            for e in set(edges):
                neighbours = self.get_pair_if_any(e)
                for ne in neighbours:
                    if (self.count_all_edges) or (not ne in self.covered_fuzzer_edges):
                        neighbour_bbl_id = str(((int(self.edge_to_parentBB[ne]) >> 1) ^ int(ne)) & 0xfffff)
                        try:
                            ne_score = float(self.bb_cov_map[neighbour_bbl_id])
                            if ne_score > 0:
                                is_interesting_edge = True
                                reachable += ne_score
                        except KeyError:
                            # print "warning something is wrong"
                            pass
            self.input_to_reachable_cov_cache[seed] = reachable
        return self.input_to_reachable_cov_cache[seed]

    def get_ctxt_edge_difference(self, seed, _):
        edges = set(self.input_to_edges_cache[seed])
        return len(edges - self.explored_fuzzer_edges)

    def get_undiscover_neighbour(self, seed, _):
        edges = self.input_to_edges_cache[seed]
        neighbours = set()
        for e in set(edges):
            neighbours = neighbours | set(self.get_pair_if_any(e))
        return len(neighbours)

    def get_path_length(self, seed, _):
        if not self.input_to_edges_cache.has_key(seed):
            self.batch_collect_features(seed, _)
        return len(self.input_to_edges_cache[seed])

    def get_covered_edges(self, seed, _):
        if not self.input_to_edges_cache.has_key(seed):
            self.batch_collect_features(seed, _)
        edges = self.input_to_edges_cache[seed]
        self.covered_fuzzer_edges = self.covered_fuzzer_edges | set(edges)
        return self.input_to_edges_cache[seed]

    def get_num_cmp(self, seed, _):
        if not self.input_to_num_cmp_cache.has_key(seed):
            self.batch_collect_features(seed, _)
        return self.input_to_num_cmp_cache[seed]

    def get_num_indcall(self, seed, _):
        if not self.input_to_num_indcall_cache.has_key(seed):
            self.batch_collect_features(seed, _)
        return self.input_to_num_indcall_cache[seed]

    def get_num_extcall(self, seed, _):
        if not self.input_to_num_extcall_cache.has_key(seed):
            self.batch_collect_features(seed, _)
        return self.input_to_num_extcall_cache[seed]

    def get_num_label(self, seed, _):
        if not self.input_to_num_label_cache.has_key(seed):
            self.batch_collect_features(seed, _)
        return self.input_to_num_label_cache[seed]

    def get_num_memops(self, seed, _):
        if not self.input_to_num_memops_cache.has_key(seed):
            self.batch_collect_features(seed, _)
        return self.input_to_num_memops_cache[seed]

    def pre_filter(self, seeds, covered, max_return = 500):
        ret = []
        unexplored = list(filter(lambda x: x not in covered, seeds))
        sorted_files = sorted(unexplored, key = functools.cmp_to_key(self.testcase_compare), reverse = True)
        for i in sorted_files:
            if len(ret) > max_return:
                break
            ret.append(i)
        return ret

    # `raw_data` is dummy
    def get_result(self, raw_data, max_results):

        seeds = [os.path.join(self.fuzzer_input_dir, x) for x in self.read_queue()]
        if self.queue_size is None:
            self.init_queue_size = len(seeds)
        self.queue_size = len(seeds)

        # try to update model
        self.update_model()

        oracle_info("read %d seeds" % len(seeds))
        stats = self.get_scores(raw_data, seeds)

        stats = sorted(stats, key=itemgetter('score'), reverse=True)
        # print stats
        result = {}
        for stat in stats:
            # Don't add more results than requested
            if max_results != -1 and len(result) >= max_results:
                break
            try:
                edge_ids = stat['interesting_edges']
            except KeyError:
                edge_ids = []
            try:
                block_id = stat['interesting_block']
            except KeyError:
                block_id = None
            score = stat['score']
            input_file = stat['first_seen']
            if input_file not in result:
                result[input_file] = {
                    'score': score,
                    'interesting_edges': edge_ids,
                    'interesting_blocks': [block_id],
                    'size': stat['size'],
                    'input': input_file
                }
            else :
                result[input_file]['interesting_edges'].extend(edge_ids)
                result[input_file]['interesting_blocks'].append(block_id)
        return result

    # collect #cmp, #indcall, #extcall, #label on path.
    # store the result in respective caches.
    def batch_collect_features(self, seed, _):
        ef = os.path.join(self.log_base, ".coverededges" + str(random.randint(1,100000000)))
        cf =  os.path.join(self.log_base, ".cmptmp" + str(random.randint(1,100000000)))
        icf = os.path.join(self.log_base, ".indcalltmp" + str(random.randint(1,100000000)))
        ecf = os.path.join(self.log_base, ".extcalltmp" + str(random.randint(1,100000000)))
        lf =  os.path.join(self.log_base, ".sanlabeltmp" + str(random.randint(1, 100000000)))
        mopf = os.path.join(self.log_base, ".moptmp" + str(random.randint(1, 1000000000)))
        envs = {
            "AFL_LOC_TRACE_FILE" : ef,
            "CMP_LOG" : cf,
            "INDIRECT_CALL_LOG" : icf,
            "EXTERNAL_CALL_LOG" : ecf,
            "SAVIOR_LABEL_LOG"  : lf,
            "MEM_OP_LOG"        : mopf,
            "ASAN_OPTIONS"      : "detect_odr_violation=0"
        }
        utils.run_one_with_envs(self.replay_prog_cmd, seed, self.input_mode, envs, 0.2, _)
        cnum = utils.count_file_items(cf, with_weight=True)
        icnum = utils.count_file_items(icf, with_weight=True)
        ecnum = utils.count_file_items(ecf, with_weight=True)
        lnum = utils.count_file_items(lf, with_weight=True)
        mopnum = utils.count_file_items(mopf, with_weight=True)
        if os.path.exists(ef):
            self.input_to_edges_cache[seed] = utils.read_edges_from_file(ef)
            self.covered_fuzzer_edges |= set(self.input_to_edges_cache[seed])
            os.unlink(ef)
        if os.path.exists(cf):
            self.input_to_num_cmp_cache[seed] = cnum
            os.unlink(cf)
        if os.path.exists(icf):
            self.input_to_num_indcall_cache[seed] = icnum
            os.unlink(icf)
        if os.path.exists(ecf):
            self.input_to_num_extcall_cache[seed] = ecnum
            os.unlink(ecf)
        if os.path.exists(lf):
            self.input_to_num_label_cache[seed] = lnum
            os.unlink(lf)
        if os.path.exists(mopf):
            self.input_to_num_memops_cache[seed] = mopnum
            os.unlink(mopf)

    # MEUZZ interfaces
    def get_feature(self, seed, _):
        """ Return collected features in a list
        [reachable labels, path length, undiscovered neihbours, new cov, size]
        """
        if self.input_to_feature_cache.has_key(seed):
            return self.input_to_feature_cache[seed]

        feature = []

        # reachable ubsan bug
        feature.append(self.get_reachable_bug(seed, _)/RB_NORM)
        # reachable cov
        feature.append(self.get_reachable_cov(seed, _)/RC_NORM)
        # path: the smaller the better
        feature.append(self.get_path_length(seed, _)/P_NORM)
        # undiscovered neighbours
        feature.append(self.get_undiscover_neighbour(seed, _)/UN_NORM)
        # +cov
        feature.append(1 if seed.endswith("+cov") else 0)
        # size: the smaller the better
        feature.append(os.path.getsize(seed)/S_NORM)
        # number of cmps on the path
        feature.append(self.get_num_cmp(seed, _)/CN_NORM)
        # number of indirect calls on the path
        feature.append(self.get_num_indcall(seed, _)/ICN_NORM)
        # number of external calls on the path
        feature.append(self.get_num_extcall(seed, _)/ECN_NORM)
        # number of labels on the path
        feature.append(self.get_num_label(seed, _)/SLN_NORM)
        # number of seeds in queue right now
        feature.append(self.queue_size/Q_NORM)
        # number of memops defined by ASAN
        feature.append(self.get_num_memops(seed, _)/MOP_NORM)
        # coverage difference
        feature.append(self.get_ctxt_edge_difference(seed, _)/CED_NORM)

        self.input_to_feature_cache[seed] = feature
        # print "seed: ", seed
        # print "feature: ", feature

        return feature

    def get_features(self, seeds, _ = False):
        res = []
        for seed in seeds:
            res.append((seed, self.get_feature(seed, _)))
        print "Len[i2e cache]: ", len(self.input_to_edges_cache)
        print "Len[i2rb cache]: ", len(self.input_to_reachable_bug_cache)
        print "Len[i2rc cache]: ", len(self.input_to_reachable_cov_cache)
        print "Len[i2f cache]: ", len(self.input_to_feature_cache)
        print "Len[i2nc cache]: ", len(self.input_to_num_cmp_cache)
        print "Len[i2ni cache]: ", len(self.input_to_num_indcall_cache)
        print "Len[i2nl cache]: ", len(self.input_to_num_label_cache)
        print "Len[i2nm cache]: ", len(self.input_to_num_memops_cache)

        return res

    def get_labels(self, wait=5):
        """
        Measure how many new seeds are contributed by the given `seed`.
        The contribution consists of:
        1) Directly imported descendants
        2) Saved mutants of the imported descendants in the given time window

        return a list of (label, feature) tuples
        """

        res = []
        record = self.record_factory.get_record(wait)
        print "wait period:", wait
        if record is not None:
            key, features = record
            print "key: ",key
            label = self.collect_label(key)
            for feature in features:
                res.append((label, feature))
            print "get label: ", res
        else:
            print "Wait record data is not ready"
        return res

    # called everytime before we make a prediction of seed utilities
    def update_model(self):
        print "trying to update model", self.data_store_file
        # print self.data_store

        entries = self.get_labels(self.meuzz_window)
        to_dump = []
        for entry in entries:
            label, feature = entry
            label = self.normalize_label(label)
            feature = feature[1]
            data = self.data_store.add_data(label, feature)
            to_dump.append((feature, label))

            # only give positive feedback
            if feature > 0:
                print "update model", feature, label
                self.meuzz.update_model([feature], [label])

        self.data_store.dump_data(to_dump, self.data_store_file, window=self.meuzz_window)
        if len(entries) != 0:
            oracle_info("queue size: {0}".format(self.queue_size))
            oracle_info("init queue size: {0}".format(self.init_queue_size))
            self.meuzz_window += 1
            self.meuzz_window = min(self.meuzz_window, \
                                    round(utils.log2(self.queue_size/self.init_queue_size)) + 1)
            oracle_info("Adjusting meuzz window size: {0}".format(self.meuzz_window))

    # record the global edges there were explored by SEs before.
    # this is unsound since edges might be different under different paths.
    def record_explored_fuzzer_edges(self, seeds):
        for seed in seeds:
            self.explored_fuzzer_edges |= set(self.input_to_edges_cache[seed])


    # called by moriarty when a batch of seeds were selected
    def oracle_add_features(self, key, seeds):
        self.record_factory.add_records(key, self.get_features(seeds, True))
        self.record_explored_fuzzer_edges(seeds)

    def collect_label(self, key):
        return utils.count_children(self.fuzzer_input_dir, key)

class DataStore:
    """This class stores timestamp features and label"""
    def __init__(self, data_file=None, feature_columns=None):
        self.store = []
        self.dump_counter = 0
	if not os.path.exists(data_file):
	    with open(data_file, "w") as outf:
	        field_names = ['window', 'id']+ feature_columns +['label']
	        writer = csv.DictWriter(outf, fieldnames=field_names, delimiter=',')
                writer.writeheader()

    def serialize_data_records(self, data_records, filename, counter=0, window=None):
        print "saving data_store to: ",filename
        try:
            with open(filename, "a") as outf:
                writer = csv.writer(outf, delimiter=',')
                num = 0
                for data in data_records:
                    tmp = dict()
                    # print "SSS sequence number: ", counter
                    writer.writerow([window, counter] + data[0] + [data[1]] )
                    print "write record: "
                    print tmp
                    counter += 1
                    num += 1
            return num
        except Exception:
            print data_records
            utils.error_msg("can not serialize data store {0}".format(filename))
            return 0

    def add_data(self, label, features):
        self.store.append((features, label))

    #incrementally dump data while keeping track of sequence number
    def dump_data(self, data, filename=None, window=None):
        if not filename is None:
            num = self.serialize_data_records(data, filename, self.dump_counter, window=window)
            self.dump_counter += num
        else:
            print "Data:"
            print data

    def dump(self, filename=None):
        if not filename is None:
            self.serialize_data_records(self.store, filename)
        else:
            print "DataStore:"
            print self.store

class RecordFactory:
    """
    This class stores records of the features and label key
    needed to use the producer-consumer model

    Maintains a separate bookkeeping stuture to record the
    age and taken or not for each record.
    """
    def __init__(self):
        self.counter= 0
        # {counter(int) : (key(str), features of multiple seeds([[]])) }
        self.record_book = {}
        self.meta_store = []
        self.seed_record = []

    def dump(self):
        print "RecordBook:"
        print self.record_book

    def gen_counter(self):
        self.counter += 1
        return self.counter

    def get_seeds(self):
        return self.seed_record

    def record_seeds(self, seeds):
        # assume there is no duplicates
        self.seed_record.extend(seeds)

    '''
    key: str for label inference
    features: [(seed_path, [x1,x2,x3,...])]
    '''
    def add_records(self, key, features):
        print "add records: ",key, features
        index = self.gen_counter()
        self.record_book[index] = (key, features)
        self.update_meta_store(index)
        self.record_seeds([x[1] for x in features])

    def get_record(self, time):
        index = self.query_meta_store(time)
        if index > 0:
            print "self counter: ",self.counter
            print "requested window: ", time
            print "giving out record[%d]" % index
            return self.record_book[index]
        else:
            print "Cooking: records not ready"
            return None

    # producer function
    def update_meta_store(self, new_record_idx):
        # update age of existing records
        for r in self.meta_store:
            r['age'] += 1
        # construct and add new record
        record = {"age" : 0, "taken": False, "index": new_record_idx}
        self.meta_store.append(record)

    # consumer function
    def query_meta_store(self, mature):
        res = -1
        for r in self.meta_store:
            if r["taken"]:
                continue
            if r["age"] >= mature:
                r["taken"] = True
                res = r["index"]
                break
        return res
