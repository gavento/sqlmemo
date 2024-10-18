import asyncio
from dataclasses import dataclass
import datetime
import functools
import getpass
import hashlib
import inspect
import pickle
import re
import socket
import threading
from contextlib import contextmanager
from typing import Any, Callable, Generator, Iterable, Optional, Type
import warnings

from pytest import Session
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
import dill

from . import serialize
from .schema import Base, FunctionState, CachedFunctionEntry


@dataclass
class CachedFunctionStats:
    local_hits: int
    local_misses: int
    cache_size: int
    cache_done: int
    cache_running: int
    cache_error: int


class CachedFunction:

    DEFAULT_DB_URL = "sqlite:///:memory:"
    DEFAULT_TABLE_NAME_PREFIX = "cache_"
    FUNCTION_CACHE_KEY = "_sqlcache"

    def __init__(
        self,
        db: str | Engine | None = None,
        *,
        func: Callable | None = None,
        func_name: str | None = None,
        table_name: str | None = None,
        hash_factory: Callable = hashlib.sha256,
        args_pickle: bool | Callable = False,
        args_json: bool | Callable = False,
        return_json: bool | Callable = False,
        record_exceptions: bool | Iterable[Type] = False,
        reraise_exceptions: bool | Iterable[Type] = False,
        use_dill: bool = False,
        store_default_args: bool = True,
    ):
        """
        Initializes a CachedFunction object.

        The cache stores the pickled result of the function, indexed by a hash of the function arguments and the function name.

        Parameters:
        - db (str | Engine | None): The database URL or SQLAlchemy Engine object. If None, an in-memory temporary DB is used.
        - func (Callable | None): The function to be cached. If None, the function will be set later using the `__call__` method.
        - func_name (str | None): The name of the function. If None, the name will be inferred from the function object.
        - table_name (str | None): The name of the database table to store the cached results. If None, a default table name will be used.
        - hash_factory (Callable): The hash function to use for hashing the function arguments. SHA256 by default.
        - args_pickle (bool | Callable): Whether to store pickled function arguments in the cache.
        - args_json (bool | Callable): Whether to store JSON-encoded function arguments in the cache.
        - return_json (bool | Callable): Whether to store JSON-encoded function return value in the cache.
        - record_exceptions (bool | Iterable[Type]): Whether to record exceptions raised by the function.
        - reraise_exceptions (bool | Iterable[Type]): Whether to reraise exceptions previously raised by the
          function on subsequent calls with the same arguments.
        - use_dill (bool): Whether to use dill instead of pickle for serialization.
          Dill can serialize more types (lambdas, local classes, etc.) but is slower and complex objects
          serialization may be less stable across code changes and python versions.
        - store_default_args (bool): Whether to store default arguments in the cache (on by default).
          This is useful if you change the default arguments of the function and want to distinguish between calls with different defaults.
          Having it off allows e.g. adding new default arguments without invalidating existing cache records.

        The `args_*` and `return_*` parameters can be set to True or False to enable or disable the respective feature, or to a callable to transform the
        value before storing it. This is useful if you e.g. want to extract only some arguments or their features to be stored in the cache (e.g. as JSON).
        You can also split arguments to be stored as JSON vs pickled. Note that the argument values are NEVER used to match future calls, only their hashes.
        Example: `args_json=lambda args: {"x": args["x"], "y": args["y"], "z_len": len(args["z"])}, return_json=lambda r: r[0]`.

        For the `*_exceptions`, True stores/reraises all exceptions, False disables all, an iterable of types allows subclasses of the given exception types.
        If an exceprion is not reraised on a subsequent call, the computation is repeated and the new outcome is stored, overwriting the old record.

        Raises:
        - ValueError: If `reraise_exceptions` is True but `record_exceptions` is False.
        - The stored exception: If `reraise_exceptions` allows the exception and the exception is not recorded in the cache.
        """
        if callable(db) and not isinstance(db, Engine):
            raise TypeError(
                f"Expected Engine or str as `db`, got {type(db)}. Hint: use `@CachedFunction()` instead of `@CachedFunction`")

        self._func_name = func_name
        self._table_name = table_name
        self._func = None
        self._func_sig = None
        if func is not None:
            self._set_func(func)

        self._args_pickle = args_pickle
        self._args_json = args_json
        self._return_json = return_json
        self._hash_factory = hash_factory
        self._record_exceptions = record_exceptions
        self._reraise_exceptions = reraise_exceptions
        if self._reraise_exceptions and not self._record_exceptions:
            raise ValueError(
                "Can't reraise exceptions without recording them - configuration error?")

        self._lock = threading.Lock()
        self._engine: Optional[Engine] = None
        self._db_url: str
        self._db_initialized = False
        self._use_dill = use_dill
        self._store_default_args = store_default_args

        self._cache_hits = 0
        self._cache_misses = 0

        if db is None:
            db = self.DEFAULT_DB_URL
        if isinstance(db, Engine):
            self._engine = db
            self._db_url = str(db.url)  # Only as informative at this point
        else:
            self._db_url = db

    def _set_func(self, func: Callable):
        """Set the wrapped function during or after __init__ (for use as a decorator)"""
        assert self._func is None
        self._func = func
        self._func_sig = inspect.signature(self._func)
        if self._func_name is None:
            self._func_name = self._func.__qualname__
        if self._table_name is None:
            self._table_name = f"{
                self.DEFAULT_TABLE_NAME_PREFIX}{self._func_name}"

    @contextmanager
    def _get_locked_session(self) -> Generator[Session, None, None]:
        """
        Lock the mutex and return a new database session, only to be used locally.
        Use only as a context manager!
        """
        with self._lock:
            self._prepare_database()
            yield Session(self._engine)  # type: ignore

    def _prepare_database(self) -> None:
        """Create DB connection and ensure the tables exist. Assumes self._lock is held."""
        assert self._lock.locked()
        if self._engine is None:
            self._engine = sa.create_engine(self._db_url)
        if not self._db_initialized:
            Base.metadata.create_all(self._engine)
            self._db_initialized = True

    def add_index(self, *column_expressions: str, index_name: str | None = None):
        """
        Add an index to the table if it does not exist yet.

        Can be called multiple times. The default index name is derived from the expression, and index existence checked by that name.
        You may end up with multiple indexes on the same columns/expressions if you change the expression strings.

        The usual use is to use this with JSON columns, e.g. `add_index("args_json->>'x'", "return_json->2")`.
        """
        with self._get_locked_session() as session:
            assert self._engine is not None
            if index_name is None:
                index_name = f"ix_{self._table_name}_" + \
                    "_".join(column_expressions)
                index_name = re.sub("[^a-zA-Z0-9]", "_", index_name)
            sa.Index(index_name, *column_expressions).create(self._engine,
                                                             checkfirst=True)

    def _hash_obj(self, obj: Any) -> str:
        """
        Stable object hasher that can work with many standard types, iterables, dataclasses.
        """
        return serialize.hash_obj(obj, sort_keys=True, hash_factory=self._hash_factory)

    def _jsonize(self, obj: Any) -> Any:
        """Smart serializer that can work with iterables and dataclasses."""
        return serialize.jsonize(obj)

    def _dumps(self, obj: Any) -> bytes:
        """Pickle the object, using dill if enabled."""
        if self._use_dill:
            return dill.dumps(obj)
        else:
            return pickle.dumps(obj)

    def _loads(self, data: bytes) -> Any:
        """Unpickle the object, using dill if enabled."""
        if self._use_dill:
            return dill.loads(data)
        else:
            return pickle.loads(data)

    def _args_to_dict(self, args: tuple[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Create a single dict with all named arguments, optionally with default values applied.
        The extra keyword and positional arguments are preserved, named as in the function signature
        (usually `*args` and `**kwargs`).
        """
        assert self._func_sig is not None
        bound_args = self._func_sig.bind(*args, **kwargs)
        if self._store_default_args:
            bound_args.apply_defaults()
        return dict(bound_args.arguments)

    def _encode_value_helper(
        self,
        value: Any,
        param: bool | Callable,
        _json: bool = False,
        _pickle: bool = False,
    ) -> Any:
        if param is False:
            return None
        if callable(param):
            value = param(value)
        if _json:
            value = self._jsonize(value)
        if _pickle:
            value = self._dumps(value)
        return value

    def _exception_check_helper(self, exception: Exception, param: bool | Iterable[Type]) -> bool:
        if isinstance(param, bool):
            return param
        return isinstance(exception, tuple(param))

    def hash_args(self, *args, **kwargs) -> str:
        """Return the hash of the function arguments, can be used to check the cache."""
        args_dict = self._args_to_dict(args, kwargs)
        return self._hash_obj(args_dict)

    def get_record_by_arg_hash(self, args_hash: str) -> CachedFunctionEntry | None:
        """Return the database record for the given hash, if it exists."""
        with self._get_locked_session() as session:
            return session.scalars(sa.select(CachedFunctionEntry).filter_by(
                func_name=self._func_name, args_hash=args_hash)).one_or_none()

    def get_record_by_args(self, *args, **kwargs) -> CachedFunctionEntry | None:
        """Return the database record for the given arguments, if it exists."""
        return self.get_record_by_arg_hash(self.hash_args(*args, **kwargs))

    @property
    def cache_hits(self) -> int:
        """
        Return the number of cache hits.

        Note this is only recorded for this cache instance, not stored in the database.
        """
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        """
        Return the number of cache misses (wrapped function calls).

        Note this is only recorded for this cache instance, not stored in the database.
        """
        return self._cache_misses

    def get_stats(self) -> CachedFunctionStats:
        """
        Return the cache statistics.

        This requires a database query (though a fast one).
        """
        with self._get_locked_session() as session:
            # Count ids where func_name matches
            q = sa.select(sa.func.count(CachedFunctionEntry.id)
                          ).filter_by(func_name=self._func_name)
            return CachedFunctionStats(
                local_hits=self.cache_hits,
                local_misses=self.cache_misses,
                cache_size=session.scalars(q).one(),
                cache_done=session.scalars(
                    q.filter_by(state=FunctionState.DONE)).one(),
                cache_running=session.scalars(
                    q.filter_by(state=FunctionState.RUNNING)).one(),
                cache_error=session.scalars(
                    q.filter_by(state=FunctionState.ERROR)).one(),
            )

    def trim_cache(self, max_records: int = 0) -> None:
        """
        Trim the cache to the given number of records; 0 = delete all recirds.
        """
        with self._get_locked_session() as session:
            # Delete all rows where func_name matches, ordered by timestamp, except the max_records latest
            q = sa.select(CachedFunctionEntry.id).filter_by(func_name=self._func_name).order_by(
                CachedFunctionEntry.timestamp.desc()).limit(max_records)
            session.execute(sa.delete(CachedFunctionEntry).filter_by(func_name=self._func_name).where(
                ~CachedFunctionEntry.id.in_(q)))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self._func_name!r} at {self._db_url!r} ({self.cache_hits} hits, {self.cache_misses} misses)>"

    def _utctimestamp(self) -> float:
        """Return the current timestamp in UTC as a float."""
        return datetime.datetime.now(datetime.timezone.utc).timestamp()

    def _call_func_sync(self, *args, **kwargs):
        """The decorated function call, with caching and exception handling."""
        assert self._func is not None
        args_hash = self.hash_args(*args, **kwargs)
        args_dict = self._args_to_dict(args, kwargs)  # Used below

        with self._get_locked_session() as session:  # Transaction
            # Check for existing result
            entry = session.scalars(sa.select(CachedFunctionEntry).filter_by(
                func_name=self._func_name, args_hash=args_hash)).one_or_none()

            if entry is not None:
                if entry.state == FunctionState.DONE:
                    # Cache hit
                    assert entry.return_pickle is not None
                    self._cache_hits += 1
                    return self._loads(entry.return_pickle)
                elif entry.state == FunctionState.RUNNING:
                    # Computation is already running
                    warnings.warn(
                        f"Function {self._func_name} is already running with the same arguments, running again in parallel (will change in the future)")
                    # raise NotImplementedError(
                    #     "Function already running - TODO: complex waiting logic needed")
                elif entry.state == FunctionState.ERROR:
                    # Last execution returned an exception
                    if self._reraise_exceptions is not False:
                        # Check if the exception should be reraised
                        if entry.exception_pickle is None:
                            raise RuntimeError(
                                f"Exception not recorded, can't reraise from {
                                    self._table_name}(id={entry.id}) original exception string: {entry.exception_str!r}"
                            )
                        exc = self._loads(entry.exception_pickle)
                        if self._exception_check_helper(exc, self._reraise_exceptions):
                            self._cache_hits += 1
                            raise exc
                        # Otherwise just re-run the computation below
            else:
                # Record as running
                entry = CachedFunctionEntry(
                    func_name=self._func_name,
                    args_hash=args_hash,
                    args_json=self._encode_value_helper(
                        args_dict, self._args_json, _json=True),
                    args_pickle=self._encode_value_helper(
                        args_dict, self._args_pickle, _pickle=True),
                    user=getpass.getuser(),
                    hostname=socket.gethostname(),
                    timestamp=self._utctimestamp(),
                    state=FunctionState.RUNNING)
            session.add(entry)
            session.commit()
            entry_id = entry.id

        # Run the function
        exc = None
        try:
            self._cache_misses += 1
            value = self._func(*args, **kwargs)
        except Exception as e:
            exc = e

        # Record the result
        with self._get_locked_session() as session:  # Transaction
            updated_entry = session.get(CachedFunctionEntry, entry_id)
            if updated_entry is not None:
                entry = updated_entry
            else:
                pass
                # raise RuntimeError(f"Record for an already running {self._func_name} missing in the database (expected ID: {entry_id})")

            entry.runtime_seconds = self._utctimestamp() - entry.timestamp
            if exc is not None:
                if self._exception_check_helper(exc, self._record_exceptions):
                    entry.exception_pickle = self._dumps(exc)
                    entry.exception_str = str(exc)
                    entry.state = FunctionState.ERROR
                else:
                    # Do not record the exception, forget the function was ever running
                    session.delete(entry)
                    session.commit()
                    raise exc
            else:
                # Record the result
                entry.state = FunctionState.DONE
                entry.return_json = self._encode_value_helper(
                    value, self._return_json, _json=True)
                entry.return_pickle = self._encode_value_helper(
                    value, True, _pickle=True)
            session.add(entry)
            session.commit()

        if exc is not None:
            raise exc
        else:
            return value

    async def _call_func_async(self, *args, **kwargs):
        """
        Placeholder for async function support.
        """
        raise NotImplementedError("Async functions are not yet supported")

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to set the function to be cached.

        Will be able to handle async functions in the future.
        """
        self._set_func(func)
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def awrapper(*args, **kwargs):
                return await self._call_func_async(*args, **kwargs)

            awrapper.__setattr__(self.FUNCTION_CACHE_KEY, self)
            return awrapper
        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._call_func_sync(*args, **kwargs)

            wrapper.__setattr__(self.FUNCTION_CACHE_KEY, self)
            return wrapper
