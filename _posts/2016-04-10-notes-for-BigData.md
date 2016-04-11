---
layout: post
title: Notes for Big Data
tags: BigData
---

### 1. Basics

Hadoop ecosystem:
* HDFS: a distributed file-system for Hadoop
* HBase: Hadoop NoSQL database
* Hive: Hadoop data warehouse
* Pig: Data analysis high-level language
* Storm: Distributed real-time computation system
* Yarn: a resource-management platform responsible for managing computing resources in clusters and using them for scheduling of users' applications.
* MapReduce: a programming model for large scale data processing.
* ZookKeeper: Hadoop centralized configuration system

[!Hadoop V1 and V2](http://image.slidesharecdn.com/hadooparchitecture-091019175427-phpapp01/95/big-data-analytics-with-hadoop-25-638.jpg?cb=1411551606)

HDFS:

* Highly scalable
* Support parallel reading and processing of the data
* Fault toleratn and easy management

MapReduce: A MapReduce job splits a large data set into independent chunks and organizes them into key, value pairs for parallel processing.

* The Map function divides the input into ranges by the InputFormat and creates a map task for each range in the input. `map(key, value) -> list<key2, value2>`
* The Reduce function then collects the various results and combines them to answer the larger problem that the master node needs to solve. `reduce(key2, list<value2>) -> list<value3>`

YARN (Yet Another Resource Negotiator): split up two major responsibilities of JobTracker(the resource management and job scheduling/monitoring) into separate daemons: a global Resource Manger and per-application Application Master.

Hive: a SQL like query language to run queries on large volumes of data for ETL.

* Not for online transaction processing
* Not for real-time queries and low-level updates
* Ad-hoc queries, summarization, and data analysis

Pig: provide rich data structures and make transformations much easier. It translates the Pig Latin script into MapReduce.

* for ETL (Extract -> Transform -> Load)
* for preparing data for easier analysis
* for long series of data operations

Tez: is an extensible framework for building high performance batch and interactive data processing applications, coordinated by YARN in Apache Hadoop.


Tutorials:

* [Hadoop tutorial based on Hortonworks Sandbox](http://hortonworks.com/tutorials/)
* [Hadoop tutorial from Yahoo](https://developer.yahoo.com/hadoop/tutorial/)

I followed this [tutorial](http://hortonworks.com/hadoop-tutorial/learning-the-ropes-of-the-hortonworks-sandbox/) to install the sandbox for Hortonworks Data Platform.

After start the sandbox, we can open a webpage with `127.0.0.1:8888`, and get
[!HDP](https://raw.githubusercontent.com/hortonworks/tutorials/hdp/assets/learning-the-ropes-of-the-hortonworks-sandbox/sandbox_welcome_page_learning_the_ropes_sandbox.png)

Then, we can login the sandbox with:

```bash
# ssh <username>@<hostname> -p <port>
ssh root@127.0.0.1 -p 2222;
```

We can also explore the [Ambari page](https://ambari.apache.org), which is like an adimn system aimed at making Hadoop management simpler. The Ambari page is like this
[!Ambari](https://c2.staticflickr.com/2/1649/26058920560_33ab47deb0_c.jpg)

Update Ambari password: `$ ambari-admin-password-reset`


Here is the word count example implemented as a MapReduce program using the framework:

```python
# Part 1
mr = MapReduce.MapReduce()

# Part 2
def mapper(record):
    # key: document identifier
    # value: document contents
    key = record[0]
    value = record[1]
    words = value.split()
    for w in words:
      mr.emit_intermediate(w, 1)

# Part 3
def reducer(key, list_of_values):
    # key: word
    # value: list of occurrence counts
    total = 0
    for v in list_of_values:
      total += v
    mr.emit((key, total))

# Part 4
inputdata = open(sys.argv[1])
mr.execute(inputdata, mapper, reducer)
```

### 2. Hive

* [Hive Language Manual](https://cwiki.apache.org/confluence/display/Hive/LanguageManual)
* [SQL to Hive Cheat Sheet](http://hortonworks.com/wp-content/uploads/downloads/2013/08/Hortonworks.CheatSheet.SQLtoHive.pdf)

```bash
# simple code for creating a table
hive > CREATE TABLE mytable (name string, age int)
       ROW FORMAT DELIMITED
       FIELDS TERMINATED BY ';'
       STORED AS TEXTEFILE;
# for load data
hive > LOAD DATA LOCAL INPATH
       'data.csv' OVERWRITE INTO
       TABLE mytable;
# insert data
hive > INSERT INTO birthdays SELECT
       firstName, lastName, birthday FROM
       customers WHERE birthday IS NOT NULL;
```

```bash
# create a table and insert data into it
hive > CREATE TABLE age_count (name string, age int);
hive > INSERT OVERWRITE TABLE age_count
       SELECT age, COUNT(age)
       FROM mytable;
```

Hive supports subqueries only in the FROM clause

```bash
hive > SELECT total FROM (SELECT c1 + c2 AS total FROM mytable) my_query;
```

Common operations:
* See current tables: `hive> SHOW TABLES;`
* Check the schema: `hive> DESCRIBE mytable;`
* Check the table name: `hive> ALTER TABLE mytable RENAME to mt;`
* Add a column: `hive> ALTER TABLE mytable ADD COLUMNS (mycol STRING);`
* Drop a partition: `hive> ALTER TABLE mytable DROP PARTITION (age=17);`

Load data:
`LOAD DATA INPATH '/tmp/trucks.csv' OVERWRITE INTO TABLE trucks_stage;`

The file `trcuks.csv` is moved to `/apps/hive/warehouse/truck_stage` folder.

ORC(Optimized Row Columnar) file format provides a highly efficient way to sotre Hive data. Create an ORC table: `CREATE TABLE ... STORED AS ORC ...`

To use Hive command lines:

```bash
$ su hive
$ hive
# To quit
$ quit
```

* Use STORED AS TEXTFILE if the data needs to be stored as plain text files.
* Use STORED AS SEQUENCEFILE if the data needs to be compressed.
* Use STORED AS ORC if the data needs to be stored in ORC file format.
* Use ROW FORMAT SERDE for the RegEx SerDe.
* Use INPUTFORMAT and OUTPUTFORMAT in the file_format to specify the name of a corresponding InputFormat and OutputFormat class as a string literal.

Partitioned tables can be created using the PARTITIONED BY clause. A table can have one or more partition columns and a separate data directory is created for each distinct value combination in the partition columns.

The EXTERNAL keyword lets you create a table and provide a LOCATION so that Hive does not use a default location for this table.

Tables can also be created and populated by the results of a query in one create-table-as-select (CTAS) statement.

```SQL
CREATE TABLE new_key_value_store
   ROW FORMAT SERDE "org.apache.hadoop.hive.serde2.columnar.ColumnarSerDe"
   STORED AS RCFile
   AS
SELECT (key % 1024) new_key, concat(key, value) key_value_pair
FROM key_value_store
SORT BY new_key, key_value_pair;
```

The LIKE form of CREATE TABLE allows you to copy an existing table definition exactly (without copying its data).

```SQL
CREATE TABLE empty_key_value_store
LIKE key_value_store;
```

### 3. Pig

Pig Latin allows you to write a data flow that describes how data will be transformed (such as aggregate, join and sort). It can be extended using other languages such as Java and Python.

* [Pig Cheat Sheet](https://www.qubole.com/resources/cheatsheet/pig-function-cheat-sheet/?nabe=5625651078365184:1&utm_referrer=https%3A%2F%2Fwww.google.com%2F)
* [Another Pig Cheat Sheet](http://mortar-public-site-content.s3-website-us-east-1.amazonaws.com)
* [Pig Latin Basics](http://pig.apache.org/docs/r0.14.0/basic.html#filter)

Pig data types:
* Tuple: ordered set of values
* Bag: unordered collection of tuples
* Map: collection of key value pairs

```bash
# Example
logevents = LOAD 'my.log' AS (date: chararray, level: chararray, code: int, message: chararray);
severe = FILTER logevents BY (level == 'severe' AND code >= 500);
grouped = GROUP severe BY code;
STORE grouped INTO 'severevents';
```

Debugging tips:
* Use `illustate`, `explain`, and `describe`
* Use local mode to test script before running it in the cluster

Examples:

```pig
// load data from a file names geolocationusing HCatLoader()
a = LOAD 'geolocation' USING org.apache.hive.hcatalog.pig.HCatLoader();
// filter dataset
b = FILTER a BY event != 'normal';
// iterate through all the records
c = FOREACH b GENERATE driverid, event, (int) '1' AS occurance;
// group by driver id and iterate over each row
d = GROUP c BY driverid;
// add to the occurance
e = FOREACH d GENERATE GROUP AS driverid, SUM(c.occurance) AS t_occ;
g = LOAD 'drivermileage' USING org.apache.hive.hcatalog.pig.HCatLoader();
h = JION e BY driverid, g BY driverid;
final_data = FOREACH h GENERATE $0 AS driverid, $1 AS events, $3 AS totmiles, (float) $3/$1 AS riskfactor;
STORE final_data INTO 'riskfactor' USING org.apache.hive.hcatalog.pig.HCatStorer();
```

Example from Hortonworks : [Transfrom NYSE data](http://hortonworks.com/hadoop-tutorial/how-to-use-basic-pig-commands/)

```pig
\\ Define a relation with a schema
STOCK_A = LOAD '/user/maria_dev/NYSE_daily_prices_A.csv' USING PigStorage(',')
    AS (exchange:chararray, symbol:chararray, date:chararray,
open:float, high:float, low:float, close:float, volume:int, adj_close:float);
DESCRIBE STOCK_A;

\\ Defien a new relation based on existing one
B = LIMIT STOCK_A 100;
DESCRIBE B;

\\ View data relation
DUMP B;

\\ Select specific columns
C = FOREACH B GENERATE symbol, date, close;
DESCRIBE C;

\\ Store relationship data into a HDFS file
STORE C INTO 'output/C' USING PigStorage(',');

\\ Perform a join
DIV_A = LOAD 'NYSE_dividends_A.csv' using PigStorage(',')
    AS (exchange:chararray, symbol:chararray, date:chararray, dividend:float);
D = JOIN STOCK_A BY (symbol, date), DIV_A BY (symbol, date);
DESCRIBE D;

\\ Order By
E = ORDER DIV_A BY symbol, date asc;

\\ Group By
F = FILTER DIV_A BY symbol=='AZZ';
G = GROUP F BY dividend;
```

Nulls and Load Functions:

```pig
A = LOAD 'student' AS (name, age, gpa);
B = FILTER A BY name is not null;
```

```pig
A = LOAD 'data' USING MyStorage() AS (T: tuple(name:chararray, age: int));
B = FILTER A BY T == ('john', 25);
D = FOREACH B GENERATE T.name, [25#5.6], {(1, 5, 18)};
```

```pig
A = LOAD 'data' AS (f1:int, f2:int, B:bag{T:tuple(t1:int,t2:int)});
DUMP A;

(10,1,{(2,3),(4,6)})
(10,3,{(2,3),(4,6)})
(10,6,{(2,3),(4,6),(5,7)})
```

```pig
X = FOREACH A GENERATE f2, (f2==1?1:COUNT(B));

DUMP X;
(1,1L)
(3,2L)
(6,3L)
```

```pig
A = LOAD 'data' AS (f1:int, f2:int,f3:int);

DUMP A;
(1,2,3)
(4,2,1)
(8,3,4)
(4,3,3)
(7,2,5)
(8,4,3)

B = GROUP A BY f1;

DUMP B;
(1,{(1,2,3)})
(4,{(4,2,1),(4,3,3)})
(7,{(7,2,5)})
(8,{(8,3,4),(8,4,3)})
```



Hive and Pig Data Model Differences:

* Pig: All data objects exist and operated on in scirpt. Once the script is complete all data objects are deleted unless you stored them.
* Hive: Operate on Hadoop data store. Data, tables, and queires persist from query to query. All data is live compared to Pig.

### 4. MapReduce

* [mrjob document](https://pythonhosted.org/mrjob/index.html)

mrjob is the easiest route to writing Python programs that run on Hadoop. If you use mrjob, you’ll be able to test your code locally without installing Hadoop or run it on a cluster of your choice.

```python
# Word Count example

from mrjob.job import MRJob

class MRWordFrequencyCount(MRJob):

    def mapper(self, _, line):
        words = line.split()
        for word in words:
            word = unicode(word, "utf-8", errors="ignore")
            yield word.lower(), 1

    def reducer(self, key, values):
        yield key, sum(values)

if __name__ == '__main__':
    MRWordFrequencyCount.run()
```

```python
# Use MRStep to do multi-step

from mrjob.job import MRJob
from mrjob.step import MRStep
import re

WORD_REGEXP = re.compile(r"[\w']+")

class MRWordFrequencyCount(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_words,
                   reducer=self.reducer_count_words),
            MRStep(mapper=self.mapper_make_counts_key,
                   reducer = self.reducer_output_words)
        ]

    def mapper_get_words(self, _, line):
        words = WORD_REGEXP.findall(line)
        for word in words:
            word = unicode(word, "utf-8", errors="ignore") #avoids issues in mrjob 5.0
            yield word.lower(), 1

    def reducer_count_words(self, word, values):
        yield word, sum(values)

    def mapper_make_counts_key(self, word, count):
        yield '%04d'%int(count), word

    def reducer_output_words(self, count, words):
        for word in words:
            yield count, word

if __name__ == '__main__':
    MRWordFrequencyCount.run()
```

* The `mapper()` method takes a key and a value as args and yields as many key-value pairs as it likes.
* A `combiner` takes a key and a subset of the values for that key as input and returns zero or more (key, value) pairs.
* The `reduce()` method takes a key and an iterator of values and also yields as many key-value pairs as it likes.

You can use `-r inline` (the default), `-r local`, `-r hadoop`, or `-r emr`.


### 5. Spark

* [Spark Programming Guide](http://spark.apache.org/docs/latest/programming-guide.html#passing-functions-to-spark)


Apache Spark was designed to be a fast, general-purpose, easy-to-use computing platform. It extends the MapReduce model and takes it to a whole other level. The speed comes from the in-memory computations. Applications running in memory allow for much faster processing and response.

Zeppellin is a web-based notebook that enables interactive data analytics. It's like Ipython Notebook.

Spark's primary core abstraction is called a Resilient Distributed Dataset or RDD. It is a distributed collection of elements that is parallelized across the cluster. In other words, a RDD is an immutable collection of objects that is partitioned and distributed across multiple physical nodes of a YARN cluster and that can be operated in parallel.

SparkContext is the main entry point to everything Spark. It can be used to create RDDs and shared variables on the cluster.

```spark
\\ spark code for risk factor analysis
\\ import sql libraries
import org.apache.spark.sql.hive.orc._
import org.apache.spark.sql._

\\ instantiate HiveContext
val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)

\\ view list of table in Hive Warehouse
hiveContext.sql("show tables").collect.foreach(println)

\\ query tables to build spark RDD
val geolocation_temp1 = hiveContext.sql("select * from geolocation")

val drivermileage_temp1 = hiveContext.sql("select * from drivermileage")

\\ register a temporary table
geolocation_temp1.registerTempTable("geolocation_temp1")
drivermileage_temp1.registerTempTable("drivermileage_temp1")

\\ perform an iteration and a filter operation
val geolocation_temp2 = hiveContext.sql("SELECT driverid, count(driverid) occurance from geolocation_temp1  where event!='normal' group by driverid")

geolocation_temp2.registerTempTable("geolocation_temp2")

geolocation_temp2.collect.foreach(println)

val joined = hiveContext.sql("select a.driverid,a.occurance,b.totmiles from geolocation_temp2 a,drivermileage_temp1 b where a.driverid=b.driverid")

joined.registerTempTable("joined")

\\ view the results
joined.collect.foreach(println)

val risk_factor_spark=hiveContext.sql("select driverid, occurance, totmiles, totmiles/occurance riskfactor from joined")

risk_factor_spark.registerTempTable("risk_factor_spark")

risk_factor_spark.collect.foreach(println)

hiveContext.sql("create table finalresults( driverid String, occurance bigint,totmiles bigint,riskfactor double) stored as orc").toDF()

\\ write to ORC format
risk_factor_spark.write.orc("risk_factor_spark")

hiveContext.sql("load data inpath 'risk_factor_spark' into table finalresults")

hiveContext.sql("select * from finalresults")
```

Run Spark commands in shell (load Scala API) : `$ spark-shell`

* [PySpark](https://spark.apache.org/docs/0.9.0/python-programming-guide.html)
* [PySpark API Docs](http://spark.apache.org/docs/latest/api/python/)
* [Spark Progamming Guide](https://spark.apache.org/docs/0.9.0/scala-programming-guide.html)

* SparkConf: for configuring Spark
* SparkFiles: access files shipped with jobs
* SparkContext: main entry point for Spark functionality
* RDD: basic abstraction in Spark
* Broadcast: a broadcast variable that gets reused across tasks
* Accumulator: an "add-only" shared variable that task can only add values to
* StorageLevel: finer-grained cache persistence levels

```python
# Word Count example

from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf = conf)

input = sc.textFile("file:///sparkcourse/book.txt")
words = input.flatMap(lambda x: x.split())
wordCounts = words.countByValue()

for word, count in wordCounts.items():
    cleanWord = word.encode('ascii', 'ignore')
    if (cleanWord):
        print cleanWord, count

```

### Reference

* [Hortonworks Hadoop Tutorials](http://hortonworks.com/hadoop-tutorial)
* [Hive Manual](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-HiveDataDefinitionLanguage)
* [Pig Latin Basics](http://pig.apache.org/docs/r0.14.0/basic.html)
* [Spark Programming Guide](http://spark.apache.org/docs/latest/programming-guide.html#passing-functions-to-spark)

