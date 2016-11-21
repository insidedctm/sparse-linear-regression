package sparse.linear.regression;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;

import scala.Tuple2;
import scala.Tuple3;

class Split implements Function<String, String[]> {

	private static final long serialVersionUID = 1L;

	public String[] call(String s) { return s.split(","); }
}

class MakeTuples implements Function<String[], Tuple3<Integer, Integer, Double>> {
	public Tuple3<Integer, Integer, Double> call(String[] ss) {
		return new Tuple3<Integer, Integer, Double>(
				Integer.parseInt(ss[0]),
				Integer.parseInt(ss[1]),
				Double.parseDouble(ss[2])
				);
	}
}

class MakePairs implements Function<String[], Tuple2<Integer, Double>> {
	public Tuple2<Integer, Double> call(String[] ss) {
		return new Tuple2<Integer, Double>(
				Integer.parseInt(ss[0]),
				Double.parseDouble(ss[1])
				);
	}
}

class MakeSparseVectors implements Function<Tuple2<ArrayList<Integer>, ArrayList<Double>>,
											Vector> {
	public Vector call(Tuple2<ArrayList<Integer>, ArrayList<Double>> input) {
		return Vectors.sparse(5, convertIntegers(input._1()), convertDoubles(input._2()));
	}
	
	public static int[] convertIntegers(List<Integer> integers)
	{
	    int[] ret = new int[integers.size()];
	    for (int i=0; i < ret.length; i++)
	    {
	        ret[i] = integers.get(i).intValue();
	    }
	    return ret;
	}
	
	
	public static double[] convertDoubles(List<Double> doubles)
	{
	    double[] ret = new double[doubles.size()];
	    for (int i=0; i < ret.length; i++)
	    {
	        ret[i] = doubles.get(i).doubleValue();
	    }
	    return ret;
	}
	
}

class create implements Function<Tuple3<Integer, Integer, Double>, Tuple2<ArrayList<Integer>, ArrayList<Double>>> {
	public Tuple2<ArrayList<Integer>, ArrayList<Double>> call(Tuple3<Integer, Integer, Double> input) {
		ArrayList<Integer> cols = new ArrayList<Integer>();
		cols.add(input._2());
		ArrayList<Double> vals = new ArrayList<Double>();
		vals.add(input._3());
		
		return new Tuple2<ArrayList<Integer>, ArrayList<Double>>(
				cols,
				vals
				);
	}
}

class combine implements 
	Function2<Tuple2<ArrayList<Integer>, ArrayList<Double>>, 
			  Tuple3<Integer, Integer, Double>, 
			  Tuple2<ArrayList<Integer>, ArrayList<Double>>> {
	public Tuple2<ArrayList<Integer>, ArrayList<Double>> call(
			Tuple2<ArrayList<Integer>, ArrayList<Double>> head,Tuple3<Integer, Integer, Double> tail) {
		
		head._1().add(tail._2());
		head._2().add(tail._3());
		
		return head;
	}
}

class merge implements
Function2<Tuple2<ArrayList<Integer>, ArrayList<Double>>, 
		  Tuple2<ArrayList<Integer>, ArrayList<Double>>, 
		  Tuple2<ArrayList<Integer>, ArrayList<Double>>> {
	public Tuple2<ArrayList<Integer>, ArrayList<Double>> call(
			Tuple2<ArrayList<Integer>, ArrayList<Double>> a,Tuple2<ArrayList<Integer>, ArrayList<Double>> b) {
		a._1().addAll(b._1());
		a._2().addAll(b._2());
		
		return new Tuple2<ArrayList<Integer>, ArrayList<Double>>( a._1(), a._2());
	}
	
	public static <T> T[] concat(T[] first, T[] second) {
		T[] result = Arrays.copyOf(first,  first.length + second.length);
		System.arraycopy(second, 0, result, first.length, second.length);
		return result;
	}
}

public class Example {

	public static void main(String[] args) {
	    SparkConf conf = new SparkConf().setAppName("Sparse Linear Regression Example");
	    conf.setMaster("local"); // change this for clustered implementation
	    JavaSparkContext sc = new JavaSparkContext(conf);
	    sc.setLogLevel("ERROR");
	    
	    String path = "data/A.txt";
	    JavaRDD<String> rawA = sc.textFile(path);
	    
	    JavaRDD<Tuple3<Integer, Integer, Double>> tuples = rawA.
	    		map(new Split()).
	    		map(new MakeTuples()
	    	);
	    JavaPairRDD<Integer, Tuple3<Integer, Integer, Double>> pairs = 
	    		tuples.mapToPair(s -> new Tuple2(s._1(), s));
	    
	    JavaPairRDD<Integer, Tuple2<ArrayList<Integer>, ArrayList<Double>>> A = 
	    		pairs.combineByKey(new create(), new combine(), new merge());
	    
	    JavaPairRDD<Integer, Vector> Asparse = A.mapValues(new MakeSparseVectors());
	    System.out.println(Asparse.collect());
	    
	    String pathb = "data/b.txt";
	    JavaRDD<String> rawb = sc.textFile(pathb);
	    System.out.println(rawb.collect());
	    
	    JavaRDD<Tuple2<Integer, Double>> bTuples = rawb.
	    		map(new Split()).
	    		map(new MakePairs()
	    	);
	    
	    JavaPairRDD<Integer, Double> b = bTuples.mapToPair(el -> new Tuple2(el._1(), el._2()));
	    System.out.println(b.collect());
	    
	    JavaRDD<LabeledPoint> Ab = Asparse.join(b).map(el -> new LabeledPoint(el._2()._2(),el._2()._1()));
	    
	    System.out.println(Ab.collect());
	    
	    int numIterations = 10000000;
	    double stepSize = 0.1;
	    LinearRegressionModel model = LinearRegressionWithSGD.train(JavaRDD.toRDD(Ab), numIterations, stepSize);
	    
	    System.out.println(model.weights());
	    System.out.println(model.intercept());
	    
	    sc.close();

	}

}


