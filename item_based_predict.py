import argparse
from collections import Counter
import json
import math
from operator import add
import sys
import time
from collections import OrderedDict

from pyspark import SparkContext, StorageLevel


def score(user_scores, business_id, model, n, avg_scores):
    model = model.value

    # Get top n similar items
    if cf_type == "item_based":
        similarities = dict(
            sorted(
                [
                    (k, v)
                    for k, v in [
                    (
                        other_id,
                        model.get((other_id, business_id))
                        or model.get((business_id, other_id)),
                    )
                    for other_id, _ in user_scores.items()
                ]
                    if v
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:n]
        )
        if not similarities:
            return None
        nom = sum(similarities[k] * user_scores[k] for k in similarities)
        den = sum(similarities.values())
        term2 = nom / float(den) if den else 0
        return term2

    if cf_type == "user_based":
        similarities = dict(
            sorted(
                [
                    (k, v)
                    for k, v in [
                    (
                        other_id,
                        model.get((other_id, business_id))
                        or
                        model.get((business_id, other_id))
                        ,
                    )
                    for other_id, _ in user_scores.items()
                ]
                    if v
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:n]
        )
        if not similarities:
            return None

        nom = sum(similarities[k] * (user_scores[k] - avg_scores[k]) for k in similarities)
        den = sum(similarities.values())
        score_user = avg_scores[business_id]
        term2 = nom / float(den) if den else 0
        return score_user + term2

    


def score_all(user_scores, business_ids, model, n, avg_scores):
    user_scores = dict(user_scores)
    return [
        (business_id, score(user_scores, business_id, model, n, avg_scores))
        for business_id in business_ids
    ]


def predict_business(x, model, n, avg_scores):
    user_id, (user_scores, business_ids) = x
    scores = score_all(user_scores, business_ids, model, n, avg_scores)
    for business_id, score in scores:
        if score:
            yield OrderedDict([("user_id", user_id), ("business_id", business_id), ("stars", score)])

def predict_user(x, model, n, avg_scores):
    business_id, (user_scores, user_ids) = x
    scores = score_all(user_scores, user_ids, model, n, avg_scores)
    for user_id, score in scores:
        if score:
            if score > 5:
                score = 5.0
            if score < 1:
                score = 1.0
            yield OrderedDict([("user_id", user_id), ("business_id", business_id), ("stars", score)])


def main_user_based(train_file, test_file, model_file, output_file, n):
    sc = SparkContext.getOrCreate()
    start = time.time()

    with open('../resource/asnlib/publicdata/user_avg.json') as json_file:
        avg_scores = json.load(json_file)


    model = sc.broadcast(
        sc.textFile(model_file)
            .map(json.loads)
            .map(lambda d: ((d["u1"], d["u2"]), d["sim"]))
            .collectAsMap()
    )

    bu_pairs = (
        sc.textFile(test_file)
            .map(json.loads)
            .map(lambda d: (d["business_id"], d["user_id"]))
    )

    business_scores = (
        sc.textFile(train_file)
            .map(json.loads)
            .map(lambda d: (d["business_id"], (d["user_id"], d["stars"])))
    )

    output = business_scores.cogroup(bu_pairs).flatMap(lambda x: predict_user(x, model, n, avg_scores)).collect()

    with open(output_file, "w") as f:
        for i in output:
            f.write('{"user_id": "'+str(i['user_id']) + '", "business_id": "' + str(i['business_id']) + '", "stars": ' + str(i['stars']) + '}\n')
    f.close()

    print("Duration:",time.time()-start)



def main_item_based(train_file, test_file, model_file, output_file, n):

    sc = SparkContext.getOrCreate()
    with open('../resource/asnlib/publicdata/user_avg.json') as json_file:
        avg_scores = json.load(json_file)

    start = time.time()
    model = sc.broadcast(
        sc.textFile(model_file)
            .map(json.loads)
            .map(lambda d: ((d["b1"], d["b2"]), d["sim"]))
            .collectAsMap()
    )

    ub_pairs = (
        sc.textFile(test_file)
            .map(json.loads)
            .map(lambda d: (d["user_id"], d["business_id"]))
    )
    user_scores = (
        sc.textFile(train_file)
            .map(json.loads)
            .map(lambda d: (d["user_id"], (d["business_id"], d["stars"])))
    )


    output = user_scores.cogroup(ub_pairs).flatMap(lambda x: predict_business(x, model, n, avg_scores)).collect()

    with open(output_file, "w") as f:
        for i in output:
            f.write('{"user_id": "'+str(i['user_id']) + '", "business_id": "' + str(i['business_id']) + '", "stars": ' + str(i['stars']) + '}\n')
    f.close()

    print("Duration:",time.time()-start)

if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    model_file = sys.argv[3]
    output_file = sys.argv[4]
    cf_type = sys.argv[5]

    if cf_type == "item_based":
        main_item_based(
            train_file, test_file, model_file, output_file, 4
        )
    elif cf_type == 'user_based':
        main_user_based(
            train_file, test_file, model_file, output_file, 10
        )
