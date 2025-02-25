# -*- coding: utf-8 -*-

import sys
import urllib
import requests
#import numpy as np
import random
import json
from ast import literal_eval

def request_qtpl(query, title, para, url="http://127.0.0.1:8081/spo"):
    #url = "http://127.0.0.1:8081/spo" # SPO
    url = "http://127.0.0.1:8082/music" # music
    data_dict = {}
    data_dict["query"] = [query]
    data_dict["titles"] = [title]
    data_dict["paras"] = [para]
    json_str = json.dumps(data_dict)#, encoding='utf8')
    #json_str = json.dumps(data_dict, encoding='gb18030')
    result = requests.post(url, json_str)
    res_json = json.loads(result.text)
    probability = round(float(res_json.get("probability", [-1])[0]), 3)
    top_res = [query, title, para, str(probability)]
    print("\t".join(top_res))

if __name__ == '__main__':

    query_file = sys.argv[1] #query文件
    rank_ip = sys.argv[2] #服务ip
    with open(query_file) as f:
        for line in f:
            query, title, para, l  = line.strip('\n').split('\t')[0:4]
            request_qtpl(query, title, para, rank_ip)
