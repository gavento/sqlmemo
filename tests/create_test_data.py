from pathlib import Path
import pytest

from sqlalchemy import create_engine
from sqlcache.memoize import Memoize, RecordState


def create_test_data_plain(path: Path):
    """This helper creates a new database file and tests that the cache works. Kept for reference only, should not be called by tests!"""
    engine = create_engine(f"sqlite:///{path}")

    g = None

    @Memoize(
        engine, record_exceptions=True, reraise_exceptions=True, args_pickle=True, func_name="f", table_name="foo_bar"
    )
    def f(x, *args, y=42, z=None, **kwargs):
        if x == "err":
            raise ValueError("This is a test")
        nonlocal g
        return dict(x=x, args=args, kwargs=kwargs, g=g)

    f(1)
    f(1)
    f(1, z="x", q=b"hello\0")
    f(1, 2, 3, foo="bar")
    g = {1, 2, 3, 4}
    f(1.3, float("nan"), float("inf"), z=frozenset({1, 2, 4}), w=RecordState.ERROR)
    with pytest.raises(ValueError):
        f("err")

    stats = f._sqlcache.get_stats()
    assert stats.misses == 5
    assert stats.cache_error == 1
    assert stats.cache_size == 5


def create_test_data_dill(path: Path):
    """This helper creates a new database file and tests that the cache works. Kept for reference only, should not be called by tests!"""
    engine = create_engine(f"sqlite:///{path}")

    g = None

    @Memoize(
        engine, record_exceptions=True, reraise_exceptions=True, args_pickle=True, func_name="f", use_dill=True
    )
    def f(x, *args, y=42, z=None, **kwargs):
        if x == "err":
            raise ValueError("This is a test")
        nonlocal g
        return dict(x=x, args=args, kwargs=kwargs, g=g)

    class Foo:
        def __init__(self, x):
            self.x = x

        def foo(self):
            return self.x * 3

    g = b"hello\1"
    f(1)
    g = None
    f(1)
    g = lambda x: x * 2
    f(1, z="x", q=b"hello\0")
    g = Foo(1)
    f(1, 2, 3, foo="bar")
    g = Foo(2).foo
    f(1.3, float("nan"), float("inf"), z=frozenset({1, 2, 4}), w=RecordState.ERROR)
    with pytest.raises(ValueError):
        f("err")

    stats = f._sqlcache.get_stats()
    assert stats.misses == 5
    assert stats.cache_error == 1
    assert stats.cache_size == 5


def create_test_data_json(path: Path):
    """This helper creates a new database file and tests that the cache works. Kept for reference only, should not be called by tests!"""
    engine = create_engine(f"sqlite:///{path}")
    g = None

    @Memoize(
        engine,
        record_exceptions=True,
        reraise_exceptions=True,
        args_pickle=True,
        func_name="f",
        args_json=True,
        value_json=True,
    )
    def f(x, *args, y=42, z=None, **kwargs):
        if x == "err":
            raise ValueError("This is a test")
        nonlocal g
        return dict(x=x, args=args, kwargs=kwargs, g=g)

    f(1)
    f(1)
    g = {1, 2, 3}
    f(1, z="x", q=frozenset({1, 2, 3}))
    g = dict(a=1, b=True, c=None, d=1.0)
    f(1, 2, 3, foo="bar")
    with pytest.raises(ValueError):
        f("err")

    stats = f._sqlcache.get_stats()
    assert stats.misses == 4
    assert stats.cache_error == 1
    assert stats.cache_size == 4


if __name__ == "__main__":
    DATA_PATH = Path(__file__).parent / "data"
    create_test_data_plain(DATA_PATH / "test_data_plain.sqlite")
    create_test_data_dill(DATA_PATH / "test_data_dill.sqlite")
    create_test_data_json(DATA_PATH / "test_data_json.sqlite")
