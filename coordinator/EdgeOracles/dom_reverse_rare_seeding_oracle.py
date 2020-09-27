#!/usr/bin/env python
import math
import utils
import sys
import os
import csv
import ConfigParser
from utils import bcolors
from operator import itemgetter
import itertools

def oracle_info(s):
    print bcolors.HEADER+"[Edge-Oracle-Info]"+bcolors.ENDC, "{0}".format(s)

class DomRevRareSeedingOracle:
    def __init__(self, config, target_bin):
        #map[edge]={'bbl','dom_num'}, singleton, constantly updating
        self.edge_dom_map = dict()
        #map['BBID']={'DOMNUM'}, singleton, load once
        self.bb_dom_map = None
        self.config = config
        self.target_prog = target_bin
        self.target_dir = os.path.dirname(os.path.abspath(self.target_prog).split()[0])
        self.edge_dom_not_exist = set()

        self.get_oracle_config()
        self.load_bb2dom_map()


        self.global_dom = True 
        # self.global_dom = True
    def __repr__(self):
        return "dom-reverse-tfidf"

    def get_oracle_config(self):
        config = ConfigParser.ConfigParser()
        config.read(self.config)
        try:
            self.bb_to_dom_file = config.get("auxiliary info", "bbl_dom_map").replace("@target", self.target_dir)
        except Exception:
            utils.error_msg("bbl_dom_map file not found in %s"%self.target_dir)
            sys.exit(-1)


    def load_bb2dom_map(self):
        try:
            self.bb_dom_map = dict()
            with open(self.bb_to_dom_file) as b2d_file:
                reader = csv.DictReader(b2d_file, delimiter=',')
                for row in reader:
                    self.bb_dom_map[row['BBID']] = row['DOMNUM']
            oracle_info('Loading BBL to Domination Map %s'%self.bb_to_dom_file)
        except Exception:
            utils.error_msg("can't load bb_dom_map: %s"%self.bb_to_dom_file)
            sys.exit(-1)

    def get_result(self, raw_data, max_results, edge_threshold=1.0):
        total_execs = float(raw_data['0']['inputs'])
        stats = []
        for e, raw in raw_data.iteritems():
            if e == '0':
                continue
            stat = raw.copy()
            stat['edge_id'] = e
            try:
                #favor more freq, more seed and more dom
                stat['rev-tfidf'] = math.log(2+float(raw['seeds'])) * (float(raw['inputs'])/total_execs)
                stat['tfidf'] = math.log(1+float(raw['seeds'])) * math.log(total_execs/float(raw['inputs']))
            except Exception:
                utils.error_msg("[rev-tfidf computation wrong, fallback]")
                print stat
                stat['rev-tfidf'] = 0.1
            stats.append(stat)
            if self.global_dom:
                #if we want to get dom factor for all edges and calc scores based on this
                self.prep_edge_dom(stat['edge_id'], stat)
        stats = sorted(stats, key=itemgetter('rev-tfidf'), reverse=True)


        if self.global_dom:
            #this is for completeness
            top_candidate = stats
        else:
            #we only do the traceloc generation for the top 10*max tfidf results
            # top_candidate = itertools.islice(stats, max_results * 10)
            #this is for scalability
            top_candidate = stats[:max_results*10]


        for stat in top_candidate:
            if stat['edge_id'] not in self.edge_dom_not_exist:
                self.prep_edge_dom(stat['edge_id'], stat)
            try:
                stat['dom-rev-tfidf'] = stat['rev-tfidf'] * math.log(1 + float(self.edge_dom_map[stat['edge_id']]['dom_num']))
                # oracle_info("dom for %s edge exist"%e)
            except KeyError:
                #fallback to tfidf 
                self.edge_dom_not_exist.add(stat['edge_id'])
                # oracle_info("dom for %s edge not exist, fallback to tfidf"%e)
                stat['dom-rev-tfidf'] = stat['tfidf']
            #oracle_info("seed: %s, tfidf: %d, dom-tfidf: %d"%(stat['edge_id'],stat['tfidf'],stat['dom-tfidf'] ))

        top_dom_rev_tfidf = sorted(top_candidate, key=itemgetter('dom-rev-tfidf'), reverse=True)
        result = {}
        for stat in top_dom_rev_tfidf:
            edge_id = stat['edge_id']
            score = stat['dom-rev-tfidf']
            input_file = stat['first_seen']
            if input_file not in result:
                # Don't add more results than requested
                if max_results != -1 and len(result) >= max_results:
                    break
                result[input_file] = {
                    'score': score,
                    'interesting_edges': [edge_id],
                    'input': input_file
                }
            elif score >= edge_threshold * result[input_file]['score']:
                result[input_file]['interesting_edges'].append(edge_id)
        return result

    def prep_edge_dom(self, edge, row_log):
        """row_log is {'inputs','seeds', 'first_seen', 'edge_id','tfidf'} """
        if edge == '0' or row_log['first_seen'] is None:
            return

        if self.edge_dom_map.has_key(edge):
            return 

        if utils.gen_loctrace_file(self.target_prog, row_log['first_seen']):
            #be careful with the loctrace file, to avoid interferece with other sefuzz instances
            with open(self.target_dir+'/loctrace.csv') as trace_file:
                reader = csv.reader(trace_file, delimiter=',')
                for row in reader:
                    #cache all the encountered edges 
                    try:
                        bid = row[0]
                        eid = row[1]
                        if not self.bb_dom_map.has_key(bid):
                            # print "fuck bbdom map does not have dom for this bb"
                            continue
                        if self.edge_dom_map.has_key(eid):
                            continue
                        self.edge_dom_map[eid] = {
                            'bbl': bid,
                            'dom_num': self.bb_dom_map[bid]
                        }
                    except IndexError:
                        continue
            # utils.rmfile_force(self.target_dir+'./loctrace.csv', silent=True)
        else:
            utils.error_msg('[Fallback]loctrace not found Can not get bbid for edge %s, using input %s'%(edge, row_log['first_seen']))
            print row_log

