#!/usr/bin/env python
import math
from utils import bcolors
from operator import itemgetter

def oracle_info(s):
    print bcolors.HEADER+"[Edge-Oracle-Info]"+bcolors.ENDC, "{0}".format(s)

class RareOracle:
    def __init__(self):
        pass
    def __repr__(self):
        return "rareness"

    def get_result(self, raw_data, max_results, edge_threshold=0.85):
        total_execs = float(raw_data['0']['inputs'])
        stats = []
        for e, raw in raw_data.iteritems():
            try:
                stat = raw.copy()
                stat['edge_id'] = e
                stat['rareness'] = math.log(total_execs / float(raw['inputs']))
                stats.append(stat)
            except Exception:
                continue
        stats = sorted(stats, key=itemgetter('rareness'), reverse=True)
        result = {}
        for stat in stats:
            edge_id = stat['edge_id']
            score = stat['rareness']
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
