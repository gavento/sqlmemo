import asyncio
import enum
import functools
import getpass
import hashlib
import inspect
import pickle
import re
import socket
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Optional, Type

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    Column,
    Enum,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from . import serialize


class FunctionState(enum.Enum):
    RUNNING = "RUNNING"
    DONE = "DONE"
    ERROR = "ERROR"


class CachedFunction:

    DEFAULT_DB_URL = "sqlite:///:memory:"
    DEFAULT_TABLE_NAME_PREFIX = "cache_"
    FUNCTION_CACHE_KEY = "__sqlcache__"

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
    ):
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
            raise ValueError("Can't reraise exceptions without recording them - configuration error?")

        self._lock = threading.Lock()
        self._engine: Optional[Engine] = None
        self._sessionmaker: Optional[sessionmaker] = None
        self._table: Optional[Table] = None
        self._db_url: Optional[str] = None

        if db is None:
            db = self.DEFAULT_DB_URL
        if isinstance(db, Engine):
            self._engine = db
            self._db_url = db.url  # Only as informative at this point
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
            self._table_name = f"{self.DEFAULT_TABLE_NAME_PREFIX}{self._func_name}"

    @contextmanager
    def _get_locked_session(self):
        """
        Lock the mutex and return a new database session, only to be used locally.
        Use only as a context manager!
        """
        with self._lock:
            self._prepare_database()
            yield self._sessionmaker()

    def _prepare_database(self):
        """Create DB connection and ensure the tables exist. Assumes self._lock is held."""
        assert self._lock.locked()
        if self._engine is None:
            self._engine = create_engine(self._db_url)
        if self._sessionmaker is None:
            self._sessionmaker = sessionmaker(bind=self._engine)
            self._table = self._table_schema()
            self._table.metadata.create_all(self._engine)

    def _table_schema(self):
        """Only creates a Table object, does not change the DB."""

        metadata = MetaData()
        return Table(
            self._table_name,
            metadata,
            Column("id", Integer, primary_key=True),
            Column("timestamp", TIMESTAMP),
            Column("user", String, nullable=True),
            Column("hostname", String, nullable=True),
            Column("func_name", String),
            Column("args_hash", String),
            Column("args_pickle", LargeBinary, nullable=True),
            Column("args_json", JSON, nullable=True),
            Column("return_pickle", LargeBinary, nullable=True),
            Column("return_json", JSON, nullable=True),
            Column("exception_pickle", LargeBinary, nullable=True),
            Column("exception_str", Text, nullable=True),
            Column("state", Enum(FunctionState), nullable=False),
            Index(f"ix_{self._table_name}_func_name_args_hash", "func_name", "args_hash", unique=True),
        )

    def add_index(self, *column_expressions: str, index_name: str = None):
        """
        Add an index to the table if it does not exist yet.
        
        Can be called multiple times. The default index name is derived from the expression, and index existence checked by that name.
        You may end up with multiple indexes on the same columns/expressions if you change the expression strings.

        The usual use is to use this with JSON columns, e.g. `add_index("args_json->>'x'", "return_json->2")`.
        """
        with self._get_locked_session() as session:
            if index_name is None:
                index_name = f"ix_{self._table_name}_" + "_".join(column_expressions)
                index_name = re.sub("[^a-zA-Z0-9]", "_", index_name)
            Index(index_name, *column_expressions).create(self._engine, checkfirst=True)

    def _hash_obj(self, obj: Any) -> str:
        """
        Stable object hasher that can work with many standard types, iterables, dataclasses.
        """
        return serialize.hash_obj(obj, sort_keys=True, hash_factory=self._hash_factory)

    def _jsonize(self, obj: Any) -> Any:
        """Smart serializer that can work with iterables and dataclasses."""
        return serialize.jsonize(obj)

    def _args_to_dict(self, args: tuple[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Create a single dict with all named arguments with defaults applied.
        The extra keyword and positional arguments are preserved, named as in the function signature
        (usually `*args` and `**kwargs`).
        """
        bound_args = self._func_sig.bind(*args, **kwargs)
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
            value = pickle.dumps(value)
        return value

    def _exception_check_helper(self, exception: Exception, param: bool | Iterable[Type]) -> bool:
        if isinstance(param, bool):
            return param
        return isinstance(exception, tuple(param))

    def _insert_row_data(self, args_dict, args_hash, value=None, exception=None, running=False):
        if running:
            return_json = None
            return_pickle = None
            exception_pickle = None
            exception_str = None
            state = FunctionState.RUNNING
        elif exception is None:
            return_json = self._encode_value_helper(value, self._return_json, _json=True)
            return_pickle = self._encode_value_helper(value, True, _pickle=True)
            exception_pickle = None
            exception_str = None
            state = FunctionState.DONE
        else:
            return_json = None
            return_pickle = None
            exception_pickle = pickle.dumps(exception)
            exception_str = str(exception)
            state = FunctionState.ERROR

        args_json = self._encode_value_helper(args_dict, self._args_json, _json=True)
        args_pickle = self._encode_value_helper(args_dict, self._args_pickle, _pickle=True)
        return dict(
            timestamp=func.now(),
            user=getpass.getuser(),
            hostname=socket.gethostname(),
            func_name=self._func_name,
            args_hash=args_hash,
            args_pickle=args_pickle,
            args_json=args_json,
            return_pickle=return_pickle,
            return_json=return_json,
            exception_pickle=exception_pickle,
            exception_str=exception_str,
            state=state,
        )

    def _call_func(self, *args, **kwargs):
        args_dict = self._args_to_dict(args, kwargs)
        args_hash = self._hash_obj(args_dict)

        with self._get_locked_session() as session:  # Transaction

            # Check for existing result
            query = (
                self._table.select()
                .where(self._table.c.func_name == self._func_name)
                .where(self._table.c.args_hash == args_hash)
            )
            result = session.execute(query).fetchone()

            if result:
                if result.state == FunctionState.DONE:
                    # Cache hit
                    return pickle.loads(result.return_pickle)
                elif result.state == FunctionState.RUNNING:
                    # Computation is already running
                    raise NotImplementedError("TODO - complex logic needed")
                else:
                    # Last execution returned an exception

                    if self._reraise_exceptions is not False:
                        # Check if the exception should be reraised
                        if result.exception_pickle is None:
                            raise RuntimeError(
                                f"Exception not recorded, can't reraise from {self._table_name}(id={result.id})"
                            )
                        exc = pickle.loads(result.exception_pickle)
                        if self._exception_check_helper(exc, self._reraise_exceptions):
                            raise exc
                    # Otherwise just re-run the computation

            # Record as running
            self._upsert_row(session, self._insert_row_data(args_dict, args_hash, running=True))

        # Run the function
        try:
            value = self._func(*args, **kwargs)
        except Exception as e:
            if self._exception_check_helper(e, self._record_exceptions):
                with self._get_locked_session() as session:  # Transaction
                    self._upsert_row(session, self._insert_row_data(args_dict, args_hash, exception=e))
            raise e

        # Record the result
        with self._get_locked_session() as session:  # Transaction
            self._upsert_row(session, self._insert_row_data(args_dict, args_hash, value=value))
            return value

    def _upsert_row(self, session, data):
        """Insert or update a row in the database"""
        # Is the call present?
        query = (
            self._table.select()
            .where(self._table.c.func_name == data["func_name"])
            .where(self._table.c.args_hash == data["args_hash"])
        )
        result = session.execute(query).fetchone()
        if result:
            query = self._table.update().values(data).where(self._table.c.id == result.id)
        else:
            query = self._table.insert().values(data)
        session.execute(query)

    async def _call_func_async(self, *args, **kwargs):
        raise NotImplementedError("Async functions are not yet supported")

    def __call__(self, func: Callable) -> Callable:
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
                return self._call_func(*args, **kwargs)

            wrapper.__setattr__(self.FUNCTION_CACHE_KEY, self)
            return wrapper
