# -*- coding: utf-8 -*-
 
from pyspark import SparkContext, SparkConf
from math import sqrt

def computeDistance(x,y):
    return sqrt(sum([(a - b)**2 for a,b in zip(x,y)]))

def closestCluster(dist_list):
    cluster = dist_list[0][0]
    min_dist = dist_list[0][1]
    for elem in dist_list:
        if elem[1] < min_dist:
            cluster = elem[0]
            min_dist = elem[1]
    return (cluster,min_dist)

def sumList(x,y):
    return [x[i]+y[i] for i in range(len(x))]

def moyenneList(x,n):
    return [x[i]/n for i in range(len(x))]

# ([5.1, 3.5, 1.4, 0.2, 'Iris-setosa']),(0, (0, 0.866025403784438))
def getPlusProche(x,y):
    if(x[1][1]>y[1][1]):
        return y
    return x

def proche(x,y):
    if x[2]>y[2]:
        return y
    return x

def simpleKmeans(data, nb_clusters):
    clusteringDone = False
    number_of_steps = 0
    current_error = float("inf")
    # A broadcast value is sent to and saved  by each executor for further use
    # instead of being sent to each executor when needed.
    nb_elem = sc.broadcast(data.count())
    #############################
    # Select initial centroides #
    #############################
    clustproche=data.cartesian(data).filter(lambda x: x[0][0]!=x[1][0])\
            .map(lambda x: (x[0][0],(x[0],x[1], computeDistance(x[0][1][:-1],x[1][1][:-1]))))\
            .reduceByKey(lambda x, y : proche(x,y))\
            .map(lambda x: (x[1][1][0],(x[1][1],1)))\
            .reduceByKey(lambda x,y : (x[0],x[1]+y[1])).map(lambda x: (x[1][1],x[1][0])).top(nb_clusters)
    
    #(0, [4.4, 3.0, 1.3, 0.2])
    centroides = sc.parallelize(clustproche)\
              .map(lambda x: x[1])\
              .zipWithIndex()\
              .map(lambda x: (x[1],x[0][1][:-1]))
    #centroides = sc.parallelize(clustproche.takeSample('withoutReplacment',nb_clusters))\
    #          .zipWithIndex()\
    #          .map(lambda x: (x[1],x[0][1][:-1]))
    # (0, [4.4, 3.0, 1.3, 0.2])
    # In the same manner, zipWithIndex gives an id to each cluster 
    while not clusteringDone:
        
        #############################
        # Assign points to clusters #
        #############################
        
        joined = data.cartesian(centroides)
        # ((0, [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']), (0, [4.4, 3.0, 1.3, 0.2]))
        
        # We compute the distance between the points and each cluster
        dist = joined.map(lambda x: (x[0],(x[1][0], computeDistance(x[0][1][:-1], x[1][1]))))
        # ((1, [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']),(0,  0.866025403784438))
        min_dist=dist.map(lambda x : (x[0][0],(x[0][1],x[1]))).reduceByKey(lambda x,y : getPlusProche(x,y))
        #(1, ([5.1, 3.5, 1.4, 0.2, 'Iris-setosa']),(0,  0.866025403784438))
        clusters = min_dist.map(lambda x: (x[1][1][0], x[1][0][:-1]))
        # (2, [5.1, 3.5, 1.4, 0.2])
        
        count = clusters.map(lambda x: (x[0],1)).reduceByKey(lambda x,y: x+y)
        somme = clusters.reduceByKey(sumList)
        centroidesCluster = somme.join(count).map(lambda x : (x[0],moyenneList(x[1][0],x[1][1])))
        ############################
        # Is the clustering over ? #
        ############################
        # Let's see how many points have switched clusters.
        if number_of_steps > 0:
            switch = prev_assignment.join(min_dist)\
                                    .filter(lambda x: x[1][0][1][0] != x[1][1][1][0])\
                                    .count()
        else:
            switch = 150
        if switch == 0 or number_of_steps == 100:
            clusteringDone = True
            error = sqrt(min_dist.map(lambda x: x[1][1][1]).reduce(lambda x,y: x + y))/nb_elem.value
        else:
            centroides = centroidesCluster
            prev_assignment = min_dist
            number_of_steps += 1
        print("\n word ",number_of_steps)
    return (min_dist, error,  number_of_steps)
 
 
if __name__ == "__main__":
 
    conf = SparkConf().setAppName('exercice')
    sc = SparkContext(conf=conf)
 
    lines = sc.textFile("file:/home/cluster/user44/projet/iris.data.txt")
    data = lines.map(lambda x: x.split(','))\
            .map(lambda x: [float(i) for i in x[:4]]+[x[4]])\
            .zipWithIndex()\
            .map(lambda x: (x[1],x[0]))
    # zipWithIndex allows us to give a specific index to each point
    # (0, [5.1, 3.5, 1.4, 0.2, 'Iris-setosa'])
 
    clustering = simpleKmeans(data,3)
 
    clustering[0].saveAsTextFile("file:/home/cluster/user44/projet/env")
    # if you want to have only 1 file as a result, then: clustering[0].coalesce(1).saveAsTextFile("file:/home/cluster/user44/projet/otpo")
 
    print(clustering)


