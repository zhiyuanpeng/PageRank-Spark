from pyspark import SparkContext, SparkConf
from tqdm import tqdm
from operator import add
import re
conf = SparkConf().setAppName("pagerank").setMaster("local")
sc = SparkContext(conf=conf)


sw_init = 1.0/685230
filepath = "/home/zhiyuan/Documents/BIgData/FinalProj/web-BerkStan_clean.txt"
from_to = sc.textFile(filepath)
"""
links [(from,[to, to,..]),...]
"""
links = from_to.map(lambda link: (link.split("\t")[0], link.split("\t")[
    1])).groupByKey()
"""
sws [(from, sw_init), (from, sw_init)]
"""
ranks = links.map(lambda web_neighbors: (web_neighbors[0], sw_init))


def cal_add_item(urls, sw):
    dw = len(urls)
    for url in urls:
        yield url, sw / dw


for i in tqdm(range(20)):
    """
    links.join(sws): [(from,([tos], sw))]
    """
    old_ranks = ranks
    toadd = links.join(ranks).flatMap(
        lambda from_to_sw: cal_add_item(from_to_sw[1][0], from_to_sw[1][1]))
    ranks = toadd.reduceByKey(add).mapValues(lambda rank: rank * 0.85 +
                                                          0.15*sw_init)
    e = old_ranks.join(ranks).values().map(lambda old_val_new_val:
        abs(old_val_new_val[0] - old_val_new_val[1])).reduce(lambda x, y: x + y)
    print(f"error of iteration {i} is {e}")

# [('22', 0.15000023681497784)]

ranks.saveAsTextFile("/home/zhiyuan/Documents/BIgData/FinalProj/rank.txt")
print("done")




