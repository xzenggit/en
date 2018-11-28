---
layout: post
title: Notes for Postgres and psycopg2
tags: gh-pages
---

* A database can have several schemas
* A schemas can have several tables
* Unless a table is public, refer to a table by schema_name.table_name

Here's an example of connectting to a postgres database using psycopg2 python package

```python
import psycopg2

conn = psycopg2.connect(host="hostname", database="databasename", user="username", password="password")
cur = conn.cursor()
#list all tables in a database
#cur.execute("""SELECT table_name FROM information_schema.tables""")
#list all schemas in a database
cur.execute("""SELECT schema_name FROM information_schema.schemata;""")
for schema in cur.fetchall():
    print(schema)
    
# List all tables in certain schema
conn = psycopg2.connect(host="hostname", database="databasename", user="username", password="password")
cur = conn.cursor()
cur.execute("""SELECT table_name FROM information_schema.tables WHERE table_schema = 'xxxx'""")
for x in cur.fetchall():
    print(x)        
```
