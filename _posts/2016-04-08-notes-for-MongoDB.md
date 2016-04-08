---
layout: post
title: Notes for MongoDB
tags: MongoDB DataBase
---

## Notes for Data Wrangling with MongoDB from Udacity

We need to assess our data to:

* Test assumptions about: values, data types, and shape
* Identity errors or outliers
* Find missing values

CSV is lightweight:

* Each line of text is a single row
* Fileds are seperated by a delimeter
* Just the data itself
* Don't need special software
* All spreadsheet apps read/write CSV

Data Modeling in JSON

* Items may have different fields
* May have nested objects
* May have nested arrays

XML desgin goals

* Data transfer
* Easy to write code to read/write
* Document valiation 
* Human readable
* Supports a wide variety applications

Measures of data quality

* Validity: conforms to a schema
* Accuracy: conforms to gold standard
* Completeness: all records
* Consistency: matches other data
* Uniformity: same units

Blueprint for cleaning data

* Audit your data
* Create a data cleaning plan
* Execute the plan
* Manually correct

Inequality operators for MongoDB: $gt, $lt, $gte, $lte, $ne.

```Python
# Find cities with population > 2500, but <= 50000.
# Ineuqality operators also work for strings and dates.
query = {"population": {"$gt": 2500, "$lte", 50000}}
cities = db.cities.find(query)
```

```python
# Update all documents in database
city = db.cities.update({"name": "Munchen", "country": "Germany"},
						{"$set": {"isoCountryCode": "DEU"}},
						multi=True)
```

Other MongoDB operators: 

* $in: if in certain sets
* $exists: if certain value exists or not
* $all: must contain all the values 
* $and: two conditions must all satisfy

In MongoDB shell:

```
# First part is query condition, second part is projection for output format.
db.tweets.find({"entities.hashtags": {"$ne": []}}, {"entities.hashtags.text": 1, "_id": 0})
```

```
# remove all cities that don't have names
db.cities.remove({"name": {"$exists": 0}})
```

Aggregation framework in MongoDB

Example: who tweeted most?

* Group Tweents by user
* Count each users tweets
* Sort into descending order
* Select user at top

```python
# realize the above process through pipeline.
result = db.tweets.aggregate([
	{"$group": {"_id": "$user.screen_name", "count": {"$sum": 1}}},
	{"$sort": {"count": -1}}])
```

Use $project to:

* Include fields from the original document
* Insert computed fields
* Rename fields
* Craete fields that hold subdocuments

```python
# Question: who has the highest followers_to_friends ratio?
# Match operator
result = db.tweets.aggregate([
	{"$match": {"user.friends_count": {"$gt": 0},
				"user.followers_count": {"$gt": 0}}},
	{"$project": {"ratio": {"$divide": ["$user.followers_count",
	 									"$user.friends_count"]},
	 			  "screen_name": "$user.screen_name"}},
	{"$sort": {"ratio": -1}},
	{"$limit": 1}])
```

Use $unwind :

```python
result = db.tweets.aggregate([
	{"$unwind": "$entities.user_mentions"},
	{"$group": {"_id": "$user.screen_name",
				"count": {"$sum": 1}}},
	{"$sort": {"count": -1}}, 
	{"$limit": 1}])
```

$group operators: $sum, $first, $last, $max, $min, $avg.

```python
result = db.tweets.aggregate([
	{"$unwind": "$entities.hashtags"},
	{"$group": {"_id": "$entities.hastags.text",
				"retweet_avg": {"$avg": "$retwet_count"}}},
	{"$sort": {"retweet_avg": -1}}, 
	{"$limit": 1}])
```

Array operators: $push, $addToSet

```python
result = db.tweets.aggregate([
	{"$unwind": "$entities.hashtags"},
	{"$group": {"_id": "$user.screen_name",
				"unique_hashtags": {"$addToSet": "$entities.hashtags.text"}}},
	{"$sort": {"_id": -1}}, 
```

More aggregation operators can be found [here](https://docs.mongodb.org/manual/reference/operator/aggregation/).







