import argparse
from collections import Counter
import json
import math
from operator import add, itemgetter
import re
import random
import glob
import shutil
import time
from collections import OrderedDict
from pyspark import SparkContext, StorageLevel
import sys
import itertools

MAX_NUM = 999999999
sc = SparkContext.getOrCreate()

def combine(business_pairs, business_reviews):
    def swap(x):
        b1, (b2, rs) = x
        return b2, (b1, rs)

    def get_avegage(lst):
        d={}
        for i in lst:
            key,val = i
            d.setdefault(key, []).append(val)
        for k,v in d.items():
            d[k] = sum(v)/float(len(v))
        return d

    def reshape(x):
        b2, ((b1, rs1), rs2) = x
        rs1 = get_avegage(list(rs1))
        rs2 = get_avegage(list(rs2))
        return (b1, b2), (rs1,rs2)

    return (
        business_pairs.join(business_reviews).map(swap).join(business_reviews).map(reshape)
    )


def pearson(xy):
    x, y = xy

    common = set(x.keys()) & set(y.keys())

    if not common:
        return 0

    n = len(common)

    mean_x = sum(x[k] for k in common) / (n)
    mean_y = sum(y[k] for k in common) / (n)

    nom = sum((x[k] - mean_x) * (y[k] - mean_y) for k in common)
    sd_x = math.sqrt(sum(math.pow(x[k] - mean_x, 2) for k in common))
    sd_y = math.sqrt(sum(math.pow(y[k] - mean_y, 2) for k in common))

    den = sd_x * sd_y
    return nom / den if den else 0


def model_record_to_json(x):
    (b1, b2), p = x
    return json.dumps(OrderedDict([("b1",b1),("b2",b2),("sim",p)]))

def user_model_record_to_json(x):
    (b1, b2), p = x
    return json.dumps(OrderedDict([("u1",b1),("u2",b2),("sim",p)]))

def hash_function(indices,random_coefficients,busi_len):
    return {
        i: h
        for i, h in enumerate(
            [
                min((((i+1)*a + b*(i+1))) % busi_len for i in indices)
                for a,b in random_coefficients
            ]
        )
        if h
    }


def key_distance(x, y):
    xs = set(x.items())
    ys = set(y.items())
    isize = len(xs.intersection(ys))
    usize = len(xs.union(ys))
    return isize / float(usize)

def main_user_based(train_file, model_file):
    start = time.time()
     
    n_functions = 20
    #hash_prime = 2038074743
    random.seed(1)
    random_coefficients = [
        (random.randint(1, 2 ** 32), random.randint(1, 2 ** 32))
        for _ in range(n_functions)
    ]
    
    raw_data = sc.textFile(train_file).map(json.loads)
    busi_len = len(raw_data.map(lambda x: x['business_id']).distinct().collect())
    
    
    business_ids = sc.broadcast(
        {
            x: i
            for i, x in enumerate(
                raw_data.map(lambda d: d["business_id"]).distinct().collect()
            )
        }
    )

    user_hashes = (
        raw_data.map(lambda d: (d["user_id"], business_ids.value[d["business_id"]]))
        .groupByKey()
        .mapValues(
            lambda indices: hash_function(indices,random_coefficients,busi_len)
        )
        .flatMap(lambda x: ((kv, x) for kv in x[1].items())) #applying LSH with r = 1,b =11
    )

    user_reviews = sc.broadcast(
        raw_data.map(
            lambda d: (d["user_id"], (business_ids.value[d["business_id"]], d["stars"]))
        )
            .groupByKey()
            .mapValues(dict)
            .collectAsMap()
    )


    similar_users = (
        user_hashes.join(user_hashes)
            .values()
            .map(
            lambda xy: ((xy[0][0], xy[1][0]), key_distance(xy[0][1], xy[1][1]) >= 0.01)
        )
            .filter(lambda x: x[1])
            .keys()
            .distinct()
    )

    model = (
        similar_users.map(
            lambda xy: (xy, (user_reviews.value[xy[0]], user_reviews.value[xy[1]]))
        )
            .mapValues(pearson)
            .filter(lambda x: x[1] > 0)
            .map(user_model_record_to_json)
    )

    model.saveAsTextFile('task3_userbased_model')

    with open(model_file, 'wb') as outfile:
        for filename in glob.glob('task3_userbased_model/part*'):
            with open(filename, 'rb') as readfile:
                shutil.copyfileobj(readfile, outfile)
    shutil.rmtree('task3_userbased_model')

    print('Duration:',time.time() - start)


def main_item_based(train_file, model_file):

    start = time.time()
    raw_data = sc.textFile(train_file).map(json.loads)
    user_business = raw_data.map(lambda d: (d["user_id"], d["business_id"])).distinct()

    business_pairs = (
        user_business.join(user_business)
            .values()
            .filter(lambda x: (x[0] > x[1]))
            .map(lambda x: (x, 1))
            .reduceByKey(add)
            .filter(lambda x: x[1] >= 3)
            .keys()
    )

    business_reviews = (
        raw_data.map(lambda d: (d["business_id"], (d["user_id"], d["stars"])))
            .groupByKey()
            .cache()
    )

    model = (
        combine(business_pairs, business_reviews)
            .mapValues(pearson)
            .filter(lambda x: x[1] > 0)
            .map(model_record_to_json)
    )

    model.saveAsTextFile('task3_itembased_model')

    with open(model_file, 'wb') as outfile:
        for filename in glob.glob('task3_itembased_model/part*'):
            with open(filename, 'rb') as readfile:
                shutil.copyfileobj(readfile, outfile)

    shutil.rmtree('task3_itembased_model')

    print('Duration:',time.time()- start)


if __name__ == "__main__":
    input_file = sys.argv[1]
    model_file = sys.argv[2]
    cf_type = sys.argv[3]

    if cf_type == "item_based":
        main_item_based(input_file, model_file)
    elif cf_type == "user_based":
        main_user_based(input_file, model_file)
