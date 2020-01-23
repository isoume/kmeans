import scala.math.sqrt;
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scala.math.{pow => carre};

type AllDistType=((Array[Double], String), (Long, Double));
type TypeResult = (RDD[(Long, ((Array[Double], String), (Long, Double)))], Double, Int);
type TypeResultAssign = RDD[(Long, (Long, Double))];
type TypeOfminDist = RDD[(Long, ((Array[Double], String), (Long, Double)))];
type TypeOfJointData = ((Long, (Array[Double], String)), (Long, (Array[Double], String)), Double); 
type TypeData = RDD[(Long, (Array[Double], String))];


def sqr(x: Double): Double ={
    return carre(x,2);
}

def computeDistance(x :Array[Double], y :Array[Double]) : Double = {
    var elem=x.zip(y).map(t=> carre((t._1-t._2),2));
    return elem.sum;
}
        
def getPlusProche( x: AllDistType , y : AllDistType): AllDistType={
    if(x._2._2> y._2._2){
        return y;
    }
    return x;
}
        
def sumList(x :Array[Double], y :Array[Double]): Array[Double]={
    var el = x.zip(y).map(t=> t._1 +t._2);
    return el;
}
        
def moyenneList(x :Array[Double], n:Int): Array[Double] ={
    return x.map(t=> (t/n))
}
        
def proche(x: TypeOfJointData,y: TypeOfJointData): TypeOfJointData={
    if (x._3>y._3){
         return y
    }
    return x
}
        
def simpleKmeans(data :TypeData, nb_clusters :Int):TypeResult= {
    var clusteringDone = false;
    var number_of_steps = 0;
    var prev_assignment: TypeResultAssign=null;
    var min_dist:TypeOfminDist=null ;
    var switch=1-2;
    var error= 0.0;
    //current_error = float("inf")
    val nb_elem = sc.broadcast(data.count())
    val clus1=data.cartesian(data)
    val clust2= clus1.filter(x => x._1._1!=x._2._1).map( x => (x._1._1,(x._1,x._2, computeDistance(x._1._2._1,x._2._2._1)))).reduceByKey((x, y) => proche(x,y)).map( x => (x._2._2._1,(x._2._2,1))).reduceByKey((x,y) => (x._1,x._2+y._2)).map( x=> (x._2._2,x._2._1)).sortByKey(false).take(nb_clusters)
    var centroides = sc.parallelize(clust2).map(x => x._2).zipWithIndex().map(x => (x._2,(x._1._2._1)))

    while(clusteringDone==false){
        val joined = data.cartesian(centroides)
        //((0, ([5.1, 3.5, 1.4, 0.2], 'Iris-setosa')), (0, ([4.4, 3.0, 1.3, 0.2]))
        // We compute the distance between the points and each cluster
        val dist = joined.map( x => (x._1,(x._2._1, computeDistance(x._1._2._1, x._2._2))))
        // ((id, ([5.1, 3.5, 1.4, 0.2], 'Iris-setosa')),(centrecluster, 0.866025403784438))
        val dist_all=dist.map(x => (x._1._1,(x._1._2,x._2)))
        //((id, (([5.1, 3.5, 1.4, 0.2], 'Iris-setosa'),(centrecluster, 0.866025403784438)))
        min_dist = dist_all.reduceByKey((x , y) => getPlusProche(x,y))
        // (0, (([5.1, 3.5, 1.4, 0.2], 'Iris-setosa'), (2, 0.5385164807134504)))
        /*
        ############################################
        # Compute the new centroid of each cluster #
        ############################################
        */
        var assignment= min_dist
        var clusters = min_dist.map(x => (x._2._2._1, x._2._1._1))
        // (2, [5.1, 3.5, 1.4, 0.2])
        var count = clusters.map(x => (x._1,1)).reduceByKey((x,y)=> x+y)
        var somme = clusters.reduceByKey(sumList)
        var centroidesCluster = somme.join(count).map(x => (x._1,moyenneList(x._2._1,x._2._2)))
        /*############################
        # Is the clustering over ? #
        ############################
        # Let's see how many points have switched clusters.
        */
        print(" voic l'itération ",number_of_steps)
        if(number_of_steps > 0){
            switch = prev_assignment.join(assignment.map(x => (x._1,x._2._2))).filter(x => x._2._1._1 != x._2._2._1).count().toInt
        }
        else{
            switch = 150
        }

        if((switch == 0) || (number_of_steps >= 100)){
            clusteringDone = true
            error = sqrt(min_dist.map(x => x._2._2._2).reduce((x,y) => x + y))/nb_elem.value.toDouble
         }
        else{
            centroides = centroidesCluster
            prev_assignment = min_dist.map(x => (x._1,x._2._2))
            number_of_steps += 1
        }
     }
    return (min_dist, error, number_of_steps)
 }


val conf = SparkConf().setAppName('exercice')
val sc = SparkContext(conf=conf)
        
val lines = sc.textFile("file:/home/cluster/user44/projet/iris.data.txt")
val data = lines.map(x => x.split(',')).map( x => (Array(x(0).toDouble,x(1).toDouble,x(2).toDouble,x(3).toDouble ),x(4))).zipWithIndex().map(x => (x._2,x._1))
// zipWithIndex allows us to give a specific index to each point
//(0, [5.1, 3.5, 1.4, 0.2, 'Iris-setosa'])

val clustering = simpleKmeans(data,3)
        
val testOk= clustering._1.saveAsTextFile("file:/home/cluster/user44/scalaLast/dev")
//if you want to have only 1 file as a result, then: clustering[0].coalesce(1).saveAsTextFile("file:/home/cluster/user44/scala/testOk")
        
print(clustering)
