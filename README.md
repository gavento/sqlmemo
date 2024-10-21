# SQLMemo: Memoizing SQL-backed Computation Cache

[![PyPI - Version](https://img.shields.io/pypi/v/sqlmemo)](https://pypi.org/project/sqlmemo/) [![GitHub branch status](https://img.shields.io/github/checks-status/gavento/sqlmemo/main)](https://github.com/gavento/sqlmemo/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/gavento/sqlmemo/graph/badge.svg?token=AtpsvIAOnt)](https://codecov.io/gh/gavento/sqlmemo)

SQLMemo is a powerful and flexible persistent memoization cache for Python, primarily intended for small- to medium-scale scientific computing, and backed by an SQL database. It provides a simple yet robust way to cache function results, improving performance for expensive computations and enabling efficient data reuse across multiple runs.

## Key Features

- SQL-backed storage (SQLite by default, supports other databases)
- Rich argument signature support (primitives, dicts, lists, sets, tuples, dataclasses, good handling of `*args` and `**kwargs`)
- Flexible return value storage (anything pickle-able, with optional dill support)
- Optional argument storage (only hashes by default), optional JSONified argument and/or result storage (e.g. for programmatic or external SQL access to the records), all with filtering and transformation options (see the API below)
- Databases stable between Python versions (tested with Python 3.10-3.13)
- Thread-safe, can be used from multiple processes and machines
- Optional exception storage and optional re-raising of stored exceptions
- Simplicity, only a single dependency (SQLAlchemy), well-tested and properly typed

## Quick Start

Install SQLMemo using pip:

```bash
pip install sqlmemo
```

Here's a basic example of how to use SQLMemo:

```python
from sqlmemo import SQLMemo
import time

# Any URI supported by SQLAlchemy, uses SQLite by default
# Multiple functions can be cached in the same database
@SQLMemo("./cache.sqlite")
def slow_function(x, y=1.0, *args):
    time.sleep(1.0)
    return (x, y, *args)

# First call: takes about 1 second
result1 = slow_function("a", "b", "c")
print(result1)  # ('a', 'b', 'c')

# Second call: returns immediately from cache
result2 = slow_function("a", "b", "c")
print(result2)  # ('a', 'b', 'c')

# Access the SQLMemo instance itself
cache = slow_function._sqlmemo

# Check cache statistics
print(cache.stats)  # InstanceStats(hits=1, misses=1, errors=0)

# Get database statistics
print(cache.get_db_stats())  # DBStats(records=1, records_done=1, records_running=0, records_error=0)

# Trim the database to keep only the most recent 10 records for this function
cache.trim(keep_records=10)

# Get the database entry for the call above
record = cache.get_record(x="a", y="b", args=("c",))
print(record)  # SQLMemoRecord(id=1, func_name='slow_function', args_hash='...', ...
```

## Motivation

As a scientist and programmer, I often found myself needing to cache expensive computations across multiple runs of my experiments. While many simple caching solutions exist, they often fell short in one way or another. Some weren't thread-safe, others didn't scale well, used slow or less-portable storage (e.g. JSON files or `shelve`), and many used naive hashing methods for arguments.

I created SQLMemo to address these issues and provide a robust, flexible caching solution for scientific computing, though it may hopefully be able to help you with any general caching in your Python code. It's designed to be simple to use, yet powerful enough to handle complex caching scenarios. By open-sourcing SQLMemo, I hope to provide a trustworthy and efficient tool that other researchers and developers can rely on for their own projects.

## SQLMemo may not be the right tool for you

- ... if your cache many small and fast calls, and you need a very high-throughput cache
- ... if you need to integrate it with other job-scheduling systems or workflows (though it can be made to work for a distributed computation, these systems usually have their own caching logic)
- ... if you need to cache very large artifacts (say 10 MBs or more, though local SQLite is surprisingly good at this) or storing external files (e.g. trained ML models)

## API Reference

### `sqlmemo.SQLMemo`

Construct a new SQLMemo cache, which stores the pickled result of the function calls in a SQL database, indexed by a hash of the function arguments and the function name.

#### Parameters

- db (str | Path | Engine | None): The database URL, path, or SQLAlchemy Engine object. If None, an in-memory temporary DB is used. If a Path or a str without a scheme is provided, SQLite is assumed.
- func_name (str | None): The name of the function. If None, the name will be inferred from the qualified name of the function passed to the decorator.
- table_name (str | None): The name of the database table to store the memoized results. Defaults to "sqlmemo_data".
- store_args_pickle (bool | Iterable[str] | Callable): Whether and how to store pickled function arguments in the cache (see below).
- store_args_json (bool | Iterable[str] | Callable): Whether and how to store JSON-encoded function arguments in the cache (see below).
- store_value_json (bool | Callable): Whether and how to store JSON-encoded function return value in the cache (see below).
- store_exceptions (bool | Iterable[Type] | Callable[[Exception], bool]): Which exceptions to store from the wrapped function when called (see below).
- reraise_exceptions (bool | Iterable[Type] | Callable[[Exception], bool]): Which exceptions to reraise from previously recorded exceptions in the cache (see below).
- hash_factory (Callable): The hash function to use for hashing the function arguments. SHA256 by default.
- use_dill (bool): Whether to use dill instead of pickle for serialization. Dill can serialize more types (lambdas, local classes, etc.) but is slower and complex objects serialization may be less stable across code changes and python versions.
- apply_default_args (bool): Whether to apply default arguments to the function call before hashing and storage (True by default). True is the more defensive setting (e.g. sensitive to changing argument defaults or adding new arguments). It does allow e.g. for adding a default value to an argument where it was already called with the default.

#### Storing arguments and return values

While the cache always stores the pickled result of the function calls, it can also optionally store the function arguments as JSON and/or pickle, and can store the function return value as JSON. All of the three optional fields can be further customized to:

- Store only a subset of arguments
- Transform the arguments or the return value before storage, e.g. to extract only some arguments or their features to be stored in the cache in a more accessible format (JSON)

The values of the parameters can be:
- True: store the value as is.
- False: do not store the value.
- Callable: transform the value using the callable before storage. The callable is called with exactly the same arguments as the wrapped function for argument transformation, and with the return value (single argument) for the return value transformation.
- Iterable[str]: store only the values of the arguments with the given names (not applicable for `store_value_json`).

The JSON versions of the arguments and return values may be e.g. useful to access the SQL database with a query, e.g. `SELECT * FROM sqlmemo_data WHERE args_json->>'x' = 'value'`.

Note that the stored argument values are NEVER used to match future calls, only the stored hashes.

#### Storing and reraising exceptions

Any exception thrown by the wrapped function can be optionally stored in the cache, and optionally reraised on a subsequent call.
These features are controlled by `store_exceptions` and `reraise_exceptions` parameters, and act independently: storing an exception does not mean it will be reraised, and vice versa. When a stored exception is found without `reraise_exceptions`, the computation is repeated and the new value (or exception, with `store_exceptions` enabled) is stored, overwriting the old record.

The values of the parameters can be:
- True: store/reraise all exceptions.
- False: do not store/reraise any exceptions.
- Iterable[Type]: store/reraise exceptions that are subclasses of any of the given types. Example: `store_exceptions=(ValueError, TypeError)`.
- Callable[[Exception], bool]: store/reraise exceptions for which the callable returns True. Example: `store_exceptions=lambda e: isinstance(e, ValueError)`.

Note that exceptions raised in the called wrapped function are propagated either way, regardless of `store_exceptions` or `reraise_exceptions`.

#### Database

If not database URL is provided to the constructor, it can be set later using `SQLMemo.set_engine(...)`. This is useful to e.g. configure the database engine based on dynamic configuration.

The connection to the database and any database structures are created lazily on the first call to the wrapper. If you want to trigger the creation of the database structures (e.g. tables) before the first call, you can use `SQLMemo.get_db_stats()`.

#### Other

The instance is fully thread-safe after construction, all user-facing functions use a mutex internally to ensure thread-safety. The wrapper can be called from multiple threads at once, and the called function may call itself recursively.

The instance provides two sets of statistics:
- `stats`: statistics of this instance of SQLMemo, including the number of hits, misses, and errors.
- `get_db_stats()`: statistics of the records in the database associated with this function.

You can access the SQLMemo instance as the `_sqlmemo` attribute of the wrapped function. Example:
```python
@SQLMemo()
def my_func(...):
    ...
my_func._sqlmemo.stats  # InstanceStats(hits=0, misses=0, errors=0)
```

### `sqlmemo.SQLMemo` methods and properties

- `get_record(*args, **kwargs) -> SQLMemoRecord | None`:
  - Returns the database record for the given arguments, if it exists. Read-only access to the database, incl checking for existing records.

- `stats` property:
  - Returns the `InstanceStats` for this instance.

- `get_db_stats() -> DBStats`:
  - Returns statistics for stored records for this function.
  - Requires a database query.

- `trim(keep_records: int = 0) -> None`:
  - Trims the DB to keep only the specified number of newest records for this function.
  - By default, deletes all records for this function.

- `__repr__() -> str`:
  - Returns a string representation of the SQLMemo instance.

## Performance Considerations

- Use SQLite for small to medium-sized caches - it is surprisingly fast even with larger binary blobs (tens of MBs or more)
- Consider using a full-fledged database for larger datasets or concurrent access
- Be mindful of serialization overhead for large objects
- Compression can help with performance and database size
  - For SQLite, consider using [sqlite-zstd](https://github.com/phiresky/sqlite-zstd) (per-row compression of selected columns with shared compression dictionaries) or [sqlite_zstd_vfs](https://github.com/mlin/sqlite_zstd_vfs) (VFS-level compression)

## Roadmap and potential extensions

- Support method calls (currently needs `self` argument to be hashable)
- Handle concurrent calls with the same arguments (currently all concurrent calls with the same arguments happen; waiting and timeout logic needed)
- Consider adding automatic trimming of old records to reduce database size (possibly with more retention policies)
- Compression of stored values (not very efficient for small data, full-DB compression should be better when available)
- Support wrapping async functions (doable, unclear if needed)

## Contributing

Contributions to SQLMemo are welcome! Please submit issues and pull requests on the project's [GitHub repository](https://github.com/gavento/sqlmemo).

## Authors

- Tomáš Gavenčiak ([@gavento](https://github.com/gavento), [ACS Research](https://acsresearch.org/))

SQLMemo is released under the MIT License. See the LICENSE file for details.

## Changelog

- 0.3.0: Refactoring into a public version. Minor API changes, renamed to sqlmemo (from sqlcache), improved testing, tooling, and documentation, `apply_default_args=True` by default.
- 0.2.0: (internal) Refactoring and renaming. DB format change, changed argument storage logic, improved testing, stats, strip async stubs.
- 0.1.0: (internal) Initial release. Exception storage and reraising, JSON arguments and return values, support concurrent and recursive calls.
