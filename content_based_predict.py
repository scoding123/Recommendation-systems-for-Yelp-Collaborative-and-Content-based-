import argparse
from collections import Counter
import json
import math
from operator import add
import re
import time
import sys
import shutil

from pyspark import SparkContext, StorageLevel


def predict(d, model):
    model = model.value
    business = model.get("b_" + d["business_id"], {})
    user = model.get("u_" + d["user_id"], {})
    similarity = cosine_similarity(business, user)
    d["sim"] = similarity
    return d


def cosine_similarity(x, y):
    xvs = math.sqrt(sum(v * v for v in x.values()))
    yvs = math.sqrt(sum(v * v for v in y.values()))
    return sum(x[k] * y.get(k, 0) for k in x) / (xvs * yvs) if xvs and yvs else 0


def main(test_file, model_file, output_file):
    SparkContext.setSystemProperty('spark.executor.memory', '4g')
    SparkContext.setSystemProperty('spark.driver.memory', '4g')
    sc = SparkContext.getOrCreate()
    start = time.time()

    sc.broadcast(sc.textFile(model_file,42).saveAsTextFile('task2_model'))
    model = sc.broadcast(sc.textFile('task2_model',42).map(json.loads).collectAsMap())
    shutil.rmtree('task2_model')

    data = sc.textFile(test_file).map(json.loads)
    profiles = data.map(lambda d: predict(d, model)).filter(lambda d: d["sim"] >= 0.01).map(json.dumps)

    json_string = profiles.reduce(lambda x, y: x + "\n" + y)

    # write your string to a file
    with open(output_file, "w") as f:
        f.write(json_string.encode("utf-8"))
    f.close()

    print("Duration:",time.time()-start)

if __name__ == "__main__":
    input_file = sys.argv[1]
    model_file = sys.argv[2]
    stopwords = sys.argv[3]
    main(input_file, model_file, stopwords)
