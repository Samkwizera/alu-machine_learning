# Databases

This directory contains database exercises for MySQL, MongoDB shell scripts, and PyMongo helpers.

## Files

- `0-create_database_if_missing.sql` through `21-div.sql`: MySQL database, table, query, trigger, procedure, and function tasks.
- `22-list_databases` through `29-delete`: MongoDB shell scripts.
- `30-all.py` through `34-log_stats.py`: PyMongo helper functions and log statistics script.

## Requirements

- MySQL
- MongoDB
- Python 3.x
- PyMongo

## Usage

Run SQL scripts with `mysql`, Mongo shell scripts with `mongo`, and Python scripts directly or through the provided main files:

```bash
cat 0-create_database_if_missing.sql | mysql -hlocalhost -uroot -p
cat 24-insert | mongo my_db
./34-log_stats.py
```
