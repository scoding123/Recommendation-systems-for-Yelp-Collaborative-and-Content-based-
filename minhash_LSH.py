from pyspark import SparkContext
import sys
import itertools
import json
from time import time
from operator import add

def get_min_val(b,data,a,m):
    val = b*data*a
    val = val%m
    return val

def calculate_min_hash(user_list):
    user_list = list(set(user_list))
    signature_matrix = []
    i=0
    while i < 70:
        signature_matrix.append(MAX_NUM)
        i += 1
    j = 0
    while j < len(user_list):
        k = 1
        while k < (71):
            signature_matrix[k-1] = min(get_min_val(k,user_list[j],a,user_count),signature_matrix[k-1])
            k+=1
        j+=1
    return signature_matrix

def data_append(data,inp1,inp2):
    data.append(inp1)
    data.extend(inp2)
    return tuple(data)

def locality_hashing(x):
    business_id = x[0]
    signature_matrix = x[1]
    business_signature = []
    i=0
    while i < 70:
        business_signature.append((data_append([],i,signature_matrix[i:i+1]),business_id))
        i+=1
    return business_signature

def jaccard_similarity(candidate_pair):
    user_1 = set(characteristic_matrix[candidate_pair[0]])
    user_2 = set(characteristic_matrix[candidate_pair[1]])
    similarity_score = float(len(user_1 & user_2))/float(len(user_1 | user_2))
    return (candidate_pair, similarity_score)


start = time()
inputFile = sys.argv[1]
outputFile = sys.argv[2]
sc = SparkContext()

#read input data
rdd = sc.textFile(inputFile).map(json.loads).map(lambda x: (x['user_id'],x['business_id']))
user_rdd = rdd.map(lambda x: x[0]).distinct()
user_count = user_rdd.count()
MAX_NUM, a = 999999999, 13

# making dictionary for user and index
user_ids = rdd.map(lambda x: x[0]).collect()
user_id_dictionary = {}
for i,j in enumerate(user_ids):
    user_id_dictionary[j] = i

# building characteristic matrix
business_user_combo = rdd.map(lambda x: (x[1], user_id_dictionary[x[0]])).cache()
business_user_combo = business_user_combo.groupByKey().cache()
characteristic_matrix = business_user_combo.mapValues(lambda x: list(set(x))).collectAsMap()

#performing min hash and locality sensitive hashing to generate candidate pairs
min_hashing = business_user_combo.mapValues(calculate_min_hash).cache()
lsh_division = min_hashing.flatMap(locality_hashing).groupByKey().map(lambda x: ((x[0],x[1]),list(x[1])))
lsh_division = lsh_division.filter(lambda x: len(x[1]) > 0).map(lambda x: (x[0][0],x[0][1]))
candidate_pairs = lsh_division.flatMap(lambda x: sorted(list(itertools.combinations(sorted(list(x[1])),2)))).distinct().cache()

#calculating jaccard similarity
jaccard_score = candidate_pairs.map(jaccard_similarity).filter(lambda x: x[1]>=0.05).collect()

#writing to output file
f = open(outputFile, 'w')
for i in (jaccard_score):
    f.write('{"b1": "'+str(i[0][0]) + '", "b2": "' + str(i[0][1]) + '", "sim": ' + str(i[1]) + '}\n')
f.close()

end = time()
elapsed_time = end - start
print("Duration: " + str(elapsed_time))
