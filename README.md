# SQL-backed computation cache

This library implements simple yet powerful persistent memoization cache.

* Stored in SQL. SQLite works great for small to medium experiments and can be queried externally.
* Use with functions with very rich argument signature and (almost) arbitrary return values:
  * Arguments need to be combination of primitive types (`int`, `float`, `str`, `bytes`, ...), `dict`s, `list`s, `set`s, `tuple`s, and anhy user-defined `dataclass`es. May include keyword ags, default values, etc.
  * Return type can be anything `pickle`-able, optionaly extended by [dill](https://github.com/uqfoundation/dill), which includes e.g. `lambda`s, local-defined classes and nested functions, etc.
* The database can store a lot more info about the records (JSONified subset of arguments, JSON of result, failed function executions, ...) and can be accessed externally, e.g. by other tools. Also supports adding secondary indexes (even on JSON fields).
* The cache is resistant to bit-rot: Uses a fixed hash function (SHA256), tested to preserve hashes between releases, does not distinguish between the same argument passed.
* Fast, thread-safe and well-tested. Optional compression. Async and parallel in the works.

## Quickstart

Install with `pip install pip install git+https://github.com/gavento/sqlcache`, or your favorite python package manager.

```python
from sqlcache import CachedFunction
import time

# By default, the cache is only in-memory. Multiple functions can share a database.
@CachedFunction("sqlite:///cache.sqlite")
def slow_fun(x, y=1.0, *zs):
  time.sleep(1.0)
  return tuple(x, y, *zs)

slow_fun("a", "c", "d", y="b") # returns ("a", "b", "c", "d") after 1s
slow_fun("a", "c", "d", y="b") # returns ("a", "b", "c", "d") immediately
# Access the CachedFunction instance for the function
cache = slow_fun._sqlcache
# Some stats (just for this instance of the cache)
assert cache.cache_hits == 1
assert cache.cache_misses == 1
# Get the number of items in the database for this function
assert cache.get_cache_size() == 1
# ... and clear it (only the data for this function)
```

## The why of sqlcache:

Over the years of coding up various scientific experiments, I often needed to call my computations or some expensive parts many times over, hence the need for persistent caching. However, the simple solutions (e.g. `shelve`, storing Pandas dataframes, CSV files, JSON lines files, ...) often had one or more problems: sensitive to changing the experimental code, did not scale for more records, not thread-safe, naive or wrong hashing of arguments, and others. So I wrote this for myself, and published it in the hope it will be useful to others.

## Roadmap

* Async function support - the underlying structure is already there, needs work and testing.
* Smart handling of parallel calls with the same arguments - the second call would idealy wait for the first one. Currently they just run in parallel.
* Cache size managemet - add LRU eviction, manual trimming to size, etc. (Currently stores all )

## Advanced features

### Storing extra information

You can store additional information about the function call and its result in the cache. This can be useful for debugging, auditing, or further processing. The following options are available:

- `args_pickle`: Whether to store pickled function arguments in the cache. Can be set to `True`, `False`, or a callable to transform the value before storing it.
- `args_json`: Whether to store JSON-encoded function arguments in the cache. Can be set to `True`, `False`, or a callable to transform the value before storing it.
- `return_json`: Whether to store JSON-encoded function return value in the cache. Can be set to `True`, `False`, or a callable to transform the value before storing it.
- `record_exceptions`: Whether to record exceptions raised by the function. Can be set to `True`, `False`, or an iterable of exception types to record.
- `reraise_exceptions`: Whether to reraise exceptions previously raised by the function on subsequent calls with the same arguments. Can be set to `True`, `False`, or an iterable of exception types to reraise.

Both `args_json` and `args_pickle` also accept a function that arbitrarily processes the arguments - you can select a subset, truncate them, add new ones, etc. The callable is given the arguments `(*args, **args)` (by default with the defalut values filled in). Note that stored arguments are not relevant for future execution, so you can store any convenient argument-dependent information. Similalrly for `return_json` - you can actually modify the value arbitrarily before storing it as JSON. (The pickled value needs to be exact, for obvious reasons).

Example usage:

```python
@CachedFunction(
  db="sqlite:///cache.sqlite",
  args_json=lambda x, y: dict(x=x, sx=str(x), foo="bar"), # Process the arguments into any JSON-like object to be stored. By default, an arg dict is stored.
  return_json=True,
  record_exceptions=True,
  reraise_exceptions=True # If a previous call returned an exception, just re-raise it without calling the function again
)
def my_function(x, y):
  if x < 0:
    raise ValueError("x must be non-negative")
  return x + y
```

In this example, the function arguments and return value are stored as JSON, and any exceptions raised by the function are recorded and reraised on subsequent calls with the same arguments.

### SQLAlchemy ORM access to the cache

You can get a SQLAlchemy ORM record CachedFunctionEntry by calling `your_cache.get_record(*args, **kwargs)` with the arguments of the function call.
For modifying the entries, you should obtain a lock and a DB session from the cache with the code below. This is a probably qute unstable API, though.

```python
with your_cache._get_locked_session() as s:
  e = your_cache.get_record("foo", answer=42)
  e.timestamp = 0.0 # Mischief!
  s.add(e)
  s.commit()
```

Or you can directly work with the SQL database. The schema is the following, usually with one table per fuction (but tables can be shared).

```sql
CREATE TABLE cached_func1(
        id INTEGER NOT NULL,
        timestamp FLOAT NOT NULL,
        func_name VARCHAR NOT NULL,
        args_hash VARCHAR NOT NULL,
        state VARCHAR(7) NOT NULL,
        user VARCHAR,
        hostname VARCHAR,
        runtime_seconds FLOAT,
        args_pickle BLOB,
        args_json JSON,
        return_pickle BLOB,
        return_json JSON,
        exception_pickle BLOB,
        exception_str TEXT,
        PRIMARY KEY (id)
)
CREATE UNIQUE INDEX ix_cached_func1_func_name_args_hash ON cached_func1 (func_name, args_hash)
CREATE INDEX ix_cached_func1_func_name_state_timestamp ON cached_func1 (func_name, state, timestamp)
```

### The details: what counts as the same invocation?

The cache is indexed by:

- The DB table name derived from qualified function name (`func.__qualname__`) or given as `table_name`. If you want more functions to share a table, just set the same tablename.
- The qualified function name (`func.__qualname__`) or given as `func_name`. If you rename or move a function and want to keep using the cache, set `func_name` and `table_name` explicitly.
- SHA256 hash of all the arguments. This means the following:
  - We recursively walk the argument structure, not all types are supported. Dictionary and set keys are sorted (i.e. ordered dictionary ordering is ignored). `__hash__` is ignored because it is not stable between runs.
  - The default argument values are applied (so for `def f(x=42)`, `f()` and `f(42)` are identical).
  - It does not matter if *declared* arguments are given by name or position (e.g. `f(42)` and `f(x=42)` are the same)
  - If your function has `*args` and/or `**kwargs` parameters, these are stored just as `args` resp `kwargs` parameters (or named however you named them).
  - The names of the parameters matter - if you rename a parameter, the calls will have different hashes.
  - If you need to know more details, the actual code is equivalent to: `b = inspect.signature(func).bind(*args, **kwargs); b.apply_defaults(); return b.arguments`

## Authors

Tomáš Gavenčiak, gavento.cz

## Licence

MIT
