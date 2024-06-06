import asyncio
import functools
import getpass
import inspect
import pickle
import re
import socket
import threading
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Type

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    Column,
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


class FunctionState(Enum):
    RUNNING = "RUNNING"
    DONE = "DONE"
    ERROR = "ERROR"


class CachedFunction:

    DEFAULT_DB_URL = "sqlite:///:memory:"
    DEFAULT_TABLE_NAME_PREFIX = "cache_"

    def __init__(
        self,
        db_url: str | None = None,
        *,
        func: Callable | None = None,
        func_name: str | None = None,
        table_name: str | None = None,
        args_pickle: bool | Callable = False,
        args_json: bool | Callable = False,
        return_json: bool | Callable = False,
        record_exceptions: bool | Iterable[Type] = False,
        reraise_exceptions: bool | Iterable[Type] = False,
        indexes: Iterable[str] = (),
    ):
        if db_url is None:
            db_url = self.DEFAULT_DB_URL
        self._db_url = db_url
        self._func_name = func_name
        self._table_name = table_name
        self._func = None
        self._func_sig = None
        if func is not None:
            self._set_func(func)

        self._args_pickle = args_pickle
        self._args_json = args_json
        self._return_json = return_json
        self._indexes = tuple(indexes)
        self._record_exceptions = record_exceptions
        self._reraise_exceptions = reraise_exceptions
        if self._reraise_exceptions and not self._record_exceptions:
            raise ValueError("Can't reraise exceptions without recording them - configuration error?")

        self._lock = threading.Lock()
        self._engine: Optional[Engine] = None
        self._Session: Optional[sessionmaker] = None
        self._table: Optional[Table] = None

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
    def _get_database_lock(self):
        """
        Lock the mutex and return a new database session, only to be used locally.
        Use only as a context manager!
        """
        with self._lock:
            self._prepare_database()
            yield self._Session()

    def _prepare_database(self):
        """Create DB connection and ensure the tables exist. Assumes self._lock is held."""
        assert self._lock.locked()
        if self._engine is None:
            self._engine = create_engine(self._db_url)
            self._Session = sessionmaker(bind=self._engine)
            self._table = self._table_schema()
            self._table.metadata.create_all(self._engine)

    def _table_schema(self):
        """Only creates a Table object, does not change the DB."""
        extra_indexes = []
        for index in self._indexes:
            if isinstance(index, str):
                index = (index,)
            index_name = re.sub("[^a-zA-Z0-9]", "_", "__".join(index))
            extra_indexes.append(Index(f"ix_{self._table_name}_{index_name}"), *index)

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
            Column("state", String),
            Index(f"ix_{self._table_name}_func_name_args_hash", "func_name", "args_hash", unique=True),
            *extra_indexes,
        )

    def _hash_obj(self, obj: Any) -> str:
        """
        Stable object hasher that can work with many standard types, iterables, dataclasses.
        """
        return serialize.hash_obj(obj, sort_keys=True)

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

    def _insert_row_operation(
        self, args_dict, args_hash, value=None, exception=None, running=False, update_on_conflict=True
    ):
        args_json = self._encode_value_helper(args_dict, self._args_json, _json=True)
        args_pickle = self._encode_value_helper(args_dict, self._args_pickle, _pickle=True)
        if running:
            return_json = None
            return_pickle = None
            exception_pickle = None
            exception_str = None
            state = FunctionState.RUNNING
        elif exception is None:
            return_json = self._encode_value_helper(value, self._return_json, _json=True)
            return_pickle = self._encode_value_helper(value, self._return_json, _pickle=True)
            exception_pickle = None
            exception_str = None
            state = FunctionState.DONE
        else:
            return_json = None
            return_pickle = None
            exception_pickle = pickle.dumps(exception)
            exception_str = str(exception)
            state = FunctionState.ERROR

        op = self._table.insert().values(
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
        if update_on_conflict:
            op = op.on_conflict_update(
                index_elements=["id"],
                set_={
                    "timestamp": func.now(),
                    "user": getpass.getuser(),
                    "hostname": socket.gethostname(),
                    "args_pickle": args_pickle,
                    "args_json": args_json,
                    "return_pickle": return_pickle,
                    "return_json": return_json,
                    "exception_pickle": exception_pickle,
                    "exception_str": exception_str,
                    "state": state,
                },
            )
        return op

    def _call_func(self, *args, **kwargs):
        args_dict = self._args_to_dict(args, kwargs)
        args_hash = self._hash_obj(args_dict)

        with self._get_database_lock() as session:
            # Check for existing result
            query = (
                self._table.select()
                .where(self._table.c.func_name == self._func_name)
                .where(self._table.c.args_hash == args_hash)
            )
            result = session.execute(query).fetchone()

            if result:
                if result.state == FunctionState.DONE:
                    return pickle.loads(result.return_pickle)
                elif result.state == FunctionState.RUNNING:
                    raise NotImplementedError("TODO - complex logic needed")
                else:
                    if result.exception_pickle is None and self._reraise_exceptions is not False:
                        raise RuntimeError(
                            f"Exception not recorded, can't reraise in {self._table_name}(id={result.id})"
                        )
                    exc = pickle.loads(result.exception_pickle)
                    if self._exception_check_helper(exc, self._reraise_exceptions):
                        raise exc
                    else:
                        # Here we should re-run the computation
                        pass

            # Record as running
            session.execute(self._insert_row_operation(args_dict, args_hash, running=True))
            session.commit()

            # Run the function
            try:
                value = self._func(*args, **kwargs)
            except Exception as e:
                if self._exception_check_helper(e, self._record_exceptions):
                    session.execute(self._insert_row_operation(args_dict, args_hash, exception=e))
                    session.commit()
                raise e

            # Record the result
            session.execute(self._insert_row_operation(args_dict, args_hash, value=value))
            session.commit()
            return value

    async def _call_func_async(self, *args, **kwargs):
        raise NotImplementedError("Async functions are not yet supported")

    def __call__(self, func: Callable) -> Callable:
        self._set_func(func)
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def awrapper(*args, **kwargs):
                return await self._call_func_async(*args, **kwargs)

            return awrapper
        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._call_func(*args, **kwargs)

            return wrapper
