#!/usr/bin/env python
import math
from utils import bcolors
from operator import itemgetter

def oracle_info(s):
    print bcolors.HEADER+"[Edge-Oracle-Info]"+bcolors.ENDC, "{0}".format(s)

class RareSeedingOracle:
    def __init__(self):
        pass

    def __repr__(self):
        return "tfidf"

    def get_result(self, raw_data, max_results, edge_threshold=1.00):
        try:
            total_execs = float(raw_data['0']['inputs'])
        except Exception:
            return {}
        stats = []
        for e, raw in raw_data.iteritems():
            try:
                stat = raw.copy()
                stat['edge_id'] = e
                stat['tfidf'] = math.log(1+float(raw['seeds'])) * math.log(total_execs/float(raw['inputs']))
                stats.append(stat)
            except Exception:
                continue
            # oracle_info("seed: %s, tfidf: %d, rareness: %d"%(stat['edge_id'],stat['tfidf'], math.log(total_execs/float(raw['inputs']) )))
        stats = sorted(stats, key=itemgetter('tfidf'), reverse=True)
        result = {}
        for stat in stats:
            edge_id = stat['edge_id']
            score = stat['tfidf']
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
