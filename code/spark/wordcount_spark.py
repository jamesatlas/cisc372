import sys, os

os.environ["PYSPARK_PYTHON"]="python3"

from pyspark import SparkContext, SparkConf

if __name__ == "__main__":

  # create Spark context with Spark configuration
  conf = SparkConf().setMaster("local[2]").setAppName("Spark Count")
  sc = SparkContext(conf=conf)
  # read in text file and split each document into words
  filename = "nonsense.txt"
  text_file = sc.textFile(filename)
  step0 = text_file.flatMap(lambda line: line.split())
  
  step1 = step0.map(lambda word: (word, 1))
  step2 = step1.reduceByKey(lambda a, b: a + b)
  step3 = step2.collect()
  
  print(step3)