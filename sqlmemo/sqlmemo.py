import asyncio
import dataclasses
import datetime
import functools
import getpass
import hashlib
import inspect
from pathlib import Path
import pickle
import re
import socket
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generator, Iterable, Optional, ParamSpec, Protocol, Type, TypeVar, cast

import sqlalchemy as sa
from pytest import Session
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from . import serialize
from .schema import SQLMemoRecord, SQLMemoState, concrete_memoize_record

P = ParamSpec("P")  # Captures the parameter types
R = TypeVar("R", covariant=True)  # Captures the return type


# Define a protocol with a callable signature and an additional '_sqlmemo' attribute
class MemoizedFn(Protocol[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

    _sqlmemo: "SQLMemo"


@dataclass
class InstanceStats:
    hits: int
    misses: int
    errors: int


@dataclass
class DBStats:
    records: int
    records_done: int
    records_running: int
    records_error: int


class SQLMemo:

    DEFAULT_DB_URL = "sqlite:///:memory:"
    DEFAULT_TABLE_NAME = "sqlmemo_data"

    def __init__(
        self,
        db: str | Path | Engine | None = None,
        *,
        func_name: str | None = None,
        table_name: str | None = None,
        store_args_pickle: bool | Iterable[str] | Callable = False,
        store_args_json: bool | Iterable[str] | Callable = False,
        store_value_json: bool | Callable = False,
        record_exceptions: bool | Iterable[Type] | Callable[[Exception], bool] = False,
        reraise_exceptions: bool | Iterable[Type] | Callable[[Exception], bool] = False,
        hash_factory: Callable = hashlib.sha256,
        use_dill: bool = False,
        apply_default_args: bool = True,
    ):
        """
        Initializes a Memoize object.

        The cache stores the pickled result of the function, indexed by a hash of the function arguments and the function name.

        Parameters:
        - db (str | Engine | None): The database URL or SQLAlchemy Engine object. If None, an in-memory temporary DB is used.
        - func_name (str | None): The name of the function. If None, the name will be inferred from the function passed to the decorator.
        - table_name (str | None): The name of the database table to store the memoized results. If None, a default table name will be used.
        - hash_factory (Callable): The hash function to use for hashing the function arguments. SHA256 by default.
        - args_pickle (bool | Callable): Whether to store pickled function arguments in the cache.
        - args_json (bool | Callable): Whether to store JSON-encoded function arguments in the cache.
        - value_json (bool | Callable): Whether to store JSON-encoded function return value in the cache.
        - record_exceptions (bool | Iterable[Type]): Whether to record exceptions from the wrapped function when called. In this case, they are propoagated either way.
        - reraise_exceptions (bool | Iterable[Type]): Whether to reraise exceptions previously recorded in the cache with the same arguments.
        - use_dill (bool): Whether to use dill instead of pickle for serialization.
          Dill can serialize more types (lambdas, local classes, etc.) but is slower and complex objects
          serialization may be less stable across code changes and python versions.
        - apply_default_args (bool): Whether to apply default arguments to the function call before hashing and storage (on by default).
          This is the mode refensive setting (e.g. sensitive to changing argument defaults or adding new arguments).
          It does allow e.g. for adding a default value to an argument where it was already called with the default.

        The `args_*` and `value_json` parameters can be set to True or False to enable or disable the respective feature, or to a callable to transform the
        value before storing it. This is useful if you e.g. want to extract only some arguments or their features to be stored in the cache (e.g. as JSON).
        You can also split arguments to be stored as JSON vs pickled. Note that the argument values are NEVER used to match future calls, only their hashes.
        Example: `args_json=lambda x, y, z: dict(x=x, sy=str(y), z_len=len(z)}, value_json=lambda r: r[0]`.

        For the `*_exceptions`, True stores/reraises all exceptions, False disables all, an iterable of types allows subclasses of the given exception types.
        If an exceprion is not reraised on a subsequent call, the computation is repeated and the new outcome is stored, overwriting the old record.

        Raises:
        - ValueError: If `reraise_exceptions` is True but `record_exceptions` is False.
        - The stored exception: If `reraise_exceptions` allows the exception and the exception is not recorded in the cache.
        """
        if callable(db) and not isinstance(db, Engine):
            raise TypeError(
                f"Expected `Engine` or `str` as `db`, got {type(db)}. Hint: use as `@SQLMemo()` instead of just `@SQLMemo`"
            )

        # General settings
        self._func_name = func_name
        self._store_args_pickle = store_args_pickle
        self._store_args_json = store_args_json
        self._store_value_json = store_value_json
        self._hash_factory = hash_factory
        self._record_exceptions = record_exceptions
        self._reraise_exceptions = reraise_exceptions
        self._apply_default_args = apply_default_args
        self._use_dill = use_dill
        if self._use_dill:
            try:
                import dill
            except ImportError:
                raise ImportError("dill is required with use_dill=True, but not installed")

        # Instance attributes
        self._func = None
        self._func_sig = None
        self._lock = threading.Lock()
        self._instance_stats = InstanceStats(hits=0, misses=0, errors=0)

        # Database attributes
        if table_name is None:
            table_name = self.DEFAULT_TABLE_NAME
        self._table_name = table_name
        self._engine: Optional[Engine] = None
        self._db_url: str | None = None
        if db is not None:
            self.set_engine(db)
        self._db_initialized = False
        self._db_entry_class = concrete_memoize_record(self._table_name)

    def set_engine(self, engine_or_url: Engine | str | Path):
        """
        Set the database engine or URL after __init__ but before any calls to the function.

        Useful to e.g. configure the database engine based on dynamic configuration.
        """
        if self._engine is not None:
            raise RuntimeError(f"Engine already set (to {self._engine!r})")
        if isinstance(engine_or_url, Engine):
            self._engine = engine_or_url
            self._db_url = str(self._engine.url)
            return

        db_url = str(engine_or_url)
        if "://" not in db_url:
            db_url = f"sqlite:///{db_url}"
        if db_url.startswith("sqlite://"):
            connect_args = dict(check_same_thread=False)  # We are using locks to ensure thread-safety here
            self._engine = sa.create_engine(db_url, connect_args=connect_args, poolclass=sa.StaticPool)
        else:
            self._engine = sa.create_engine(db_url)
        self._db_url = db_url

    @contextmanager
    def _get_locked_session(self) -> Generator[Session, None, None]:
        """
        Lock the instance mutex and return a new SQLAlchemy database session.

        Use only as a context manager! Be careful not to hold the session open longer than necessary.
        """
        with self._lock:
            if self._engine is None:
                self.set_engine(self.DEFAULT_DB_URL)
            if not self._db_initialized:
                self._db_entry_class.metadata.create_all(self._engine)  # type: ignore
                self._db_initialized = True
            yield Session(self._engine)  # type: ignore

    def _add_index(self, *column_expressions: str, index_name: str | None = None):
        """
        Add an index to the table if it does not exist yet.

        Can be called multiple times. By default index name is derived from the expression, and index existence checked by that name,
        you may end up with multiple indexes on the same columns/expressions if you change the expression strings.

        The usual use is to use this with "func_name" (multiple functions share the table by default) and JSON columns,
        e.g. `add_index("func_name", "args_json->>'x'", "return_json->2")`. Note that mixing this with functions that do not
        share the same arg names is fine -- the values will simply be NULL.

        Note that this method, while fully usable, is not considered entirely stable yet.
        """
        with self._get_locked_session() as session:
            assert self._engine is not None
            if index_name is None:
                index_name = f"ix__{self._table_name}__" + "__".join(column_expressions)
                index_name = re.sub("[^a-zA-Z0-9]+", "_", index_name)
            sa.Index(index_name, *column_expressions).create(self._engine, checkfirst=True)

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
            import dill

            return dill.dumps(obj)
        else:
            return pickle.dumps(obj)

    def _loads(self, data: bytes) -> Any:
        """Unpickle the object, using dill if enabled."""
        if self._use_dill:
            import dill

            return dill.loads(data)
        else:
            return pickle.loads(data)

    def _args_to_dict(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[inspect.BoundArguments, dict[str, Any]]:
        """
        Create a single dict with all named arguments, optionally with default values applied.
        The extra positional arguments are preserved, named as in the function signature (usually `args` for `*args`).
        Kwargs are added into the returned dictionary. Returns a tuple of the BoundArguments and the dictionary.
        """
        assert self._func_sig is not None
        bound_args = self._func_sig.bind(*args, **kwargs)
        if self._apply_default_args:
            bound_args.apply_defaults()
        d = dict(bound_args.arguments)
        kwargs_name = next((p.name for p in self._func_sig.parameters.values() if p.kind == inspect.Parameter.VAR_KEYWORD), None)
        if kwargs_name is not None and kwargs_name in d:
            kw = d.pop(kwargs_name)
            d.update(kw)
        return bound_args, d

    def _encode_value_helper(
        self, value: Any, param: bool | Callable, _json: bool = False, _pickle: bool = False
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

    def _encode_args_helper(
        self, args: inspect.BoundArguments, param: bool | Callable | Iterable[str], **kwargs
    ) -> Any:
        if param is False:
            return None
        elif param is True:
            _, val = self._args_to_dict(args.args, args.kwargs)
        elif isinstance(param, Iterable):
            _, val = self._args_to_dict(args.args, args.kwargs)
            val = {k: val[k] for k in param if k in val}
        elif callable(param):
            val = param(*args.args, **args.kwargs)
        else:
            assert False, f"Invalid param type: {param!r}"
        return self._encode_value_helper(val, param=True, **kwargs)

    def _exception_check_helper(
        self, exception: Exception, param: bool | Iterable[Type] | Callable[[Exception], bool]
    ) -> bool:
        if isinstance(param, bool):
            return param
        if callable(param):
            return param(exception)
        return isinstance(exception, tuple(param))

    def _hash_args(self, *args, **kwargs) -> str:
        """Return the hash of the function arguments, can be used to check the cache."""
        _, args_dict = self._args_to_dict(args, kwargs)
        return self._hash_obj(args_dict)

    def _get_record_by_hash(self, args_hash: str) -> SQLMemoRecord | None:
        """Return the database record for the given hash, if it exists."""
        with self._get_locked_session() as session:
            return session.scalars(
                sa.select(self._db_entry_class).filter_by(func_name=self._func_name, args_hash=args_hash)
            ).one_or_none()

    def get_record(self, *args, **kwargs) -> SQLMemoRecord | None:
        """Return the database record for the given arguments, if it exists."""
        return self._get_record_by_hash(self._hash_args(*args, **kwargs))

    @property
    def stats(self) -> InstanceStats:
        """
        Return the statistics for this instance.
        """
        return dataclasses.replace(self._instance_stats)

    def get_db_stats(self) -> DBStats:
        """
        Return the statistics for stored records for this function.

        This requires a database query (though a fast one).
        """
        with self._get_locked_session() as session:
            q0 = session.query(sa.func.count(self._db_entry_class.id)).filter_by(func_name=self._func_name)
            q = sa.select(
                q0.label("total"),
                q0.filter_by(state=SQLMemoState.DONE).label("done"),
                q0.filter_by(state=SQLMemoState.RUNNING).label("running"),
                q0.filter_by(state=SQLMemoState.ERROR).label("error"),
            )
            result = session.execute(q).one()

            return DBStats(
                records=result.total,
                records_done=result.done,
                records_running=result.running,
                records_error=result.error,
            )

    def trim(self, keep_records: int = 0) -> None:
        """
        Trim the DB to the given number of newest records for this function; by default deletes all records.
        """
        with self._get_locked_session() as session:
            if keep_records == 0:
                session.execute(sa.delete(self._db_entry_class).filter_by(func_name=self._func_name))
            else:
                # Delete all rows where func_name matches, ordered by timestamp, except the keep_records latest
                q = (
                    sa.select(self._db_entry_class.id)
                    .filter_by(func_name=self._func_name)
                    .order_by(self._db_entry_class.timestamp.desc())
                    .limit(keep_records)
                )
                session.execute(
                    sa.delete(self._db_entry_class)
                    .filter_by(func_name=self._func_name)
                    .where(~self._db_entry_class.id.in_(q))
                )
            session.commit()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self._func_name!r} at {self._db_url!r} table {self._table_name!r} ({self.stats.hits} hits, {self.stats.misses} misses, {self.stats.errors} errors)>"

    def _utctimestamp(self) -> float:
        """Return the current timestamp in UTC as a float."""
        return datetime.datetime.now(datetime.timezone.utc).timestamp()

    def _new_or_update_record(
        self, record: SQLMemoRecord | None, args_hash: str, bound_args: inspect.BoundArguments, state: SQLMemoState
    ) -> SQLMemoRecord:
        """Create a new entry for the given arguments."""
        if record is None:
            record = cast(SQLMemoRecord, self._db_entry_class())
        assert self._func_name is not None
        record.func_name = self._func_name
        record.args_hash = args_hash
        record.args_json = self._encode_args_helper(bound_args, self._store_args_json, _json=True)
        record.args_pickle = self._encode_args_helper(bound_args, self._store_args_pickle, _pickle=True)
        record.user = getpass.getuser()
        record.hostname = socket.gethostname()
        record.timestamp = self._utctimestamp()
        record.state = state
        return record

    def _call_wrapper(self, *args, **kwargs):
        """The decorated function call, with caching and exception handling."""
        assert self._func is not None
        args_hash = self._hash_args(*args, **kwargs)
        bound_args, args_dict = self._args_to_dict(args, kwargs)

        # Part 1: Check the cache for existing result
        with self._get_locked_session() as session:
            entry = session.execute(
                sa.select(self._db_entry_class).filter_by(func_name=self._func_name, args_hash=args_hash)
            ).scalar_one_or_none()

            if entry is not None and entry.state == SQLMemoState.DONE:
                # Cache hit
                assert entry.value_pickle is not None
                self._instance_stats.hits += 1
                return self._loads(entry.value_pickle)
            if entry is not None and entry.state == SQLMemoState.RUNNING:
                # Computation is already running, just run again in parallel
                warnings.warn(
                    f"Function {self._func_name} is already running with the same arguments, running again in parallel (will change in the future)"
                )
                # TODO: complex waiting logic needed to avoid parallel runs (the blocking logic and timeouts)
            if entry is not None and entry.state == SQLMemoState.ERROR:
                # Last execution returned an exception
                # If reraising is disabled, we just ignore the exception and run the computation below
                if self._reraise_exceptions is not False:
                    if entry.exception_pickle is None:
                        # This is unexpected, we should always have an exception for ERROR states
                        raise RuntimeError(
                            f"Exception not recorded for {self._func_name} in an ERROR-state record. Possibly a bug."
                        )
                    exc = self._loads(entry.exception_pickle)
                    if self._exception_check_helper(exc, self._reraise_exceptions):
                        self._instance_stats.hits += 1
                        raise exc
                    # Otherwise just re-run the computation below

            self._instance_stats.misses += 1
            # Record as running
            entry = self._new_or_update_record(entry, args_hash, bound_args, SQLMemoState.RUNNING)
            session.add(entry)
            session.commit()
            entry_id = entry.id
            start_time = entry.timestamp

        # Part 2: Run the function
        exc = None
        try:
            value = self._func(*args, **kwargs)
        except Exception as e:
            exc = e

        # Part 3: Record the result
        with self._get_locked_session() as session:
            entry = session.get(self._db_entry_class, entry_id)
            if entry is None:
                # The entry may have been deleted in the meantime (e.g by a trim)
                entry = self._new_or_update_record(None, args_hash, bound_args, SQLMemoState.RUNNING)

            entry.timestamp = self._utctimestamp()
            entry.runtime_seconds = entry.timestamp - start_time

            if exc is not None:
                self._instance_stats.errors += 1
                if self._exception_check_helper(exc, self._record_exceptions):
                    entry.exception_pickle = self._dumps(exc)
                    entry.exception_str = str(exc)
                    entry.state = SQLMemoState.ERROR
                    session.add(entry)
                else:
                    # Do not record the exception, forget the function was ever running
                    session.delete(entry)
                # Propagate the exception either way
                session.commit()
                raise exc

            # Record the result
            entry.state = SQLMemoState.DONE
            entry.value_json = self._encode_value_helper(value, self._store_value_json, _json=True)
            entry.value_pickle = self._encode_value_helper(value, True, _pickle=True)
            session.add(entry)
            session.commit()
            return value

    def __call__(self, func: Callable[P, R]) -> MemoizedFn[P, R]:
        """
        Decorator to set the function to be cached.

        Will be able to handle async functions in the future.
        """
        if self._func is not None:
            raise RuntimeError(
                "SQLMemo can only wrap one function, create a new instance of SQLMemo "
                f"(already wrapping {self._func_name!r}: {self._func!r})"
            )
        if asyncio.iscoroutinefunction(func):
            raise NotImplementedError("Wrapping async functions not yet supported")

        self._func = func
        self._func_sig = inspect.signature(self._func)
        if self._func_name is None:
            self._func_name = self._func.__qualname__
            if "<" in self._func_name or ">" in self._func_name:
                warnings.warn(
                    f"The decorated function qualified name {self._func_name!r} possibly not stable across runs or uses; consider setting `func_name` explicitly"
                )

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return self._call_wrapper(*args, **kwargs)

        wrapper_cast = cast(MemoizedFn[P, R], wrapper)
        wrapper_cast._sqlmemo = self
        return wrapper_cast
