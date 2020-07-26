import argparse
from collections import Counter
import json
import math
from operator import add
import re
import sys
import time
import glob
import shutil

from pyspark import SparkContext, StorageLevel


def merge(x, y):
    x.update(y)
    return x


def build_profile(reviews, key, idf):
    def top_words(d):
        tf_id = {k: v * idf[k] for k, v in d.items() if k in idf}
        return {
            k: 1
            for k, _ in sorted(tf_id.items(), key=lambda x: x[1], reverse=True)[:200]
        }

    return (
        reviews.map(lambda d: (d[key], d["tokens"]))
        .aggregateByKey(Counter(), merge, merge)
        .mapValues(top_words)
    )


def tokenize(d, stopwords, word_pattern=re.compile("\w+")):
    text = d.get("text", "").lower()
    tokens = word_pattern.findall(text)
    d["tokens"] = Counter(token for token in tokens if token not in stopwords)
    return d


def main(train_file, model_file, stopwords_file):
    SparkContext.setSystemProperty('spark.executor.memory', '4g')
    SparkContext.setSystemProperty('spark.driver.memory', '4g')
    sc = SparkContext.getOrCreate()
    start = time.time()
    stopwords = {s for s in sc.textFile(stopwords_file).collect()}
    
    reviews = (
        sc.textFile(train_file)
        .map(json.loads)
        .map(lambda d: tokenize(d, stopwords))
        .persist(StorageLevel(True, True, False, False))
    )

    n = reviews.count()
    
    #calculating number of documents the term appears in
    dfs = (
        reviews.flatMap(lambda d: d["tokens"])
        .map(lambda t: (t, 1))
        .reduceByKey(add)
        .collectAsMap()
    )

    idfs = {k: math.log(n / v) for k, v in dfs.items()}

    def add_key_prefix(rdd, prefix):
        return rdd.map(lambda x: ("{}_{}".format(prefix, x[0]), x[1]))

    business_profiles = build_profile(reviews, "business_id", idfs)

    user_profiles = (
        reviews.map(lambda d: (d["business_id"], d["user_id"]))
        .join(business_profiles)
        .values()
        .aggregateByKey({}, merge, merge)
    )


    add_key_prefix(user_profiles, "u").union(add_key_prefix(business_profiles, "b")).map(json.dumps).saveAsTextFile('task2_model')

    with open(model_file, 'wb') as outfile:
        for filename in glob.glob('task2_model/part*'):
            with open(filename, 'rb') as readfile:
                shutil.copyfileobj(readfile, outfile)
   
    shutil.rmtree('task2_model')

    print("Duration:",time.time()-start)

if __name__ == "__main__":
    input_file = sys.argv[1]
    model_file = sys.argv[2]
    stopwords = sys.argv[3]
    main(input_file, model_file, stopwords)
