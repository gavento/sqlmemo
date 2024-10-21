from collections import namedtuple
import hashlib
from pathlib import Path
import pytest

from sqlmemo.sqlmemo import SQLMemo, SQLMemoState


def create_test_data_plain(path: Path):
    """This helper creates a new database file and tests that the cache works."""

    @SQLMemo(
        path,
        record_exceptions=True,
        store_args_pickle=True,
        func_name="f",
        table_name="foo_bar",
        hash_factory=hashlib.sha3_512,
    )
    def f(x, *args, y=42, z=None, **kwargs):
        if x == "err":
            raise ValueError("This is a test")
        nonlocal g
        return (x, y, z, args, kwargs, g)

    g = None
    f(1)
    f(1, z="x", q=b"hello\0")
    f(1, 2, 3, foo="bar")
    g = {1, 2, 3, 4}
    f(1.3, float("nan"), float("inf"), z=frozenset({1, 2, 4}), w=SQLMemoState.ERROR)
    with pytest.raises(ValueError):
        f("err")

    assert f._sqlmemo.stats.misses == 5
    assert f._sqlmemo.stats.errors == 1
    assert f._sqlmemo.get_db_stats().records == 5


def create_test_data_dill(path: Path):
    """This helper creates a new database file and tests that the cache works."""

    @SQLMemo(
        path,
        record_exceptions=True,
        store_args_pickle=True,
        func_name="f",
        use_dill=True,
        apply_default_args=False,
    )
    def f(x, *args, y=42, z=None, **kwargs):
        if x == "err":
            raise ValueError("This is a test")
        nonlocal g
        return (x, y, z, args, kwargs, g)

    class Foo:
        def __init__(self, x):
            self.x = x

        def foo(self):
            return self.x * 3

    XY = namedtuple("XY", ["x", "y"])

    g = b"hello\1"
    f(1)
    g = None
    f(1)  # NB: This is not stored
    g = lambda x: x * 2
    f(1, z="x", q=b"hello\0")
    g = Foo(1)
    f(1, 2, 3, foo="bar")
    g = Foo(2).foo
    f(1.3, float("nan"), float("inf"), z=frozenset({1, 2, 4}), w=SQLMemoState.ERROR)
    with pytest.raises(ValueError):
        f("err")
    g = XY
    f("XY")  # Store the class so we can use it on the next line
    f(XY(1, 2))
    f(XY(1, 2))

    assert f._sqlmemo.stats.hits == 2
    assert f._sqlmemo.stats.misses == 7
    assert f._sqlmemo.stats.errors == 1
    assert f._sqlmemo.get_db_stats().records == 7


def create_test_data_json(path: Path):
    """This helper creates a new database file and tests that the cache works."""

    @SQLMemo(
        path,
        record_exceptions=True,
        store_args_pickle=True,
        func_name="f",
        store_args_json=True,
        store_value_json=True,
    )
    def f(x, *args, y=42, z=None, **kwargs):
        if x == "err":
            raise ValueError("This is a test")
        nonlocal g
        return (x, y, z, args, kwargs, g)

    g = None
    f(1)
    g = 1
    f(1)  # NB: This is not stored
    g = {1, 2, 3}
    f(1, z="x", q=frozenset({1, 2, 3}))
    g = dict(a=1, b=True, c=None, d=1.0)
    f(1, 2, 3, foo="bar")
    with pytest.raises(ValueError):
        g = lambda x: x * 2  # This is not even tried to be serialized to JSON
        f("err")

    assert f._sqlmemo.stats.misses == 4
    assert f._sqlmemo.stats.errors == 1
    assert f._sqlmemo.get_db_stats().records == 4


if __name__ == "__main__":
    DATA_PATH = Path(__file__).parent / "data"
    create_test_data_plain(DATA_PATH / "test_data_plain.sqlite")
    create_test_data_dill(DATA_PATH / "test_data_dill.sqlite")
    create_test_data_json(DATA_PATH / "test_data_json.sqlite")
