import inspect
import re
import threading
from typing import Callable, Any, Optional, Iterable
from sqlalchemy import (
    create_engine,
    Table,
    MetaData,
    Column,
    String,
    PickleType,
    JSON,
    TIMESTAMP,
    Text,
    Index,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import hashlib


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
        indexes: Iterable[str] = (),
    ):
        if db_url is None:
            db_url = self.DEFAULT_DB_URL
        self._db_url = db_url
        self._func_name = func_name
        self._table_name = table_name
        self._func = None
        self._func_sig = None
        # Delayed in the case of use as a decorator
        if func is not None:
            self._set_func(func)

        self._args_pickle = args_pickle
        self._args_json = args_json
        self._return_json = return_json
        self._indexes = tuple(indexes)

        self._lock = threading.Lock()
        self._engine: Optional[Engine] = None
        self._Session: Optional[sessionmaker] = None

    def _set_func(self, func: Callable):
        assert self._func is None
        self._func = func
        self._func_sig = inspect.signature(self._func)
        if self._func_name is None:
            self._func_name = self._func.__qualname__
        if self._table_name is None:
            self._table_name = f"{self.DEFAULT_TABLE_NAME_PREFIX}{self._func_name}"

    @contextmanager
    def _get_database_lock(self):
        with self._lock:
            if self._engine is None:
                self._engine = create_engine(self._db_url)
                self._Session = sessionmaker(bind=self._engine)
                self._initialize_table()
            yield self._Session()

    def _initialize_table(self):
        """Only to be called when _lock is held, and no session is open."""
        assert self._lock.locked()

        extra_indexes = []
        for index in self._indexes:
            if isinstance(index, str):
                index = (index,)
            index_name = re.sub("[^a-zA-Z0-9]", "_", "__".join(index))
            extra_indexes.append(Index(f"ix_{self._table_name}_{index_name}"), *index)

        metadata = MetaData()
        table = Table(
            self._table_name,
            metadata,
            Column("id", String, primary_key=True),
            Column("timestamp", TIMESTAMP),
            Column("user", String, nullable=True),
            Column("hostname", String, nullable=True),
            Column("func_name", String),
            Column("args_hash", String),
            Column("args_pickle", PickleType, nullable=True),
            Column("args_json", JSON, nullable=True),
            Column("return_pickle", PickleType, nullable=True),
            Column("return_json", JSON, nullable=True),
            Column("exception_pickle", PickleType, nullable=True),
            Column("exception_str", Text, nullable=True),
            Column("state", String),
            Index(
                f"ix_{self._table_name}_func_name_args_hash", "funs_name", "args_hash"
            ),
            *extra_indexes,
        )
        metadata.create_all(self._engine)

    def _hash_obj(self, obj: Any) -> str:
        """
        Smart hasher that can work with dataclasses, iterables, float NaNs, numpy arrays, etc.
        Also hashes the types of the parameters (so e.g. `(1,2)` and `[1,2]` are distinct).
        Is very stable across python versions, in particular does not use pickle or __hash__.
        Raises TypeError on an unknown type.
        """
        args_hash = hashlib.sha256()
        raise NotImplementedError()
        return args_hash.hexdigest()

    def _args_to_dict(self, *args: tuple[Any], **kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Analyse the function signature and create a single dict with all anmed arguments,
        and with `"*args"` for all the remaining arguments.
        """
        bound_args = self._func_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return dict(bound_args.arguments)

    def _dict_to_args(self, d: dict[str, Any]) -> tuple[tuple[Any], dict[str, Any]]:
        """Inverse of _args_to_dict - split `d` to named arguments and `*args`"""
        raise NotImplementedError()

    def _obj_to_json(self, obj: Any) -> str:
        """Smart serializer that can work with dataclasses, iterables, float NaNs, numpy arrays, etc."""
        raise NotImplementedError()
    
    def __call__(self, func: Callable) -> Callable:
        # TODO: wrap properly with functools
        async def awrapper(*args, **kwargs):
            return await self._call_func_async_(*args, **kwargs)

        def wrapper(*args, **kwargs):
            return self._call_func_sync(*args, **kwargs)

        self._set_func(func)
        # TODO: switch by whether the function was async or sync
        return wrapper


# Example usage:
@CachedFunction("sqlite:///cache.sqlite", args_pickle=True)
def expensive_function(x, y):
    # Expensive computation
    return x + y
