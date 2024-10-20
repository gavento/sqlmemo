import math
import os
from pathlib import Path
import pickle
import shutil
import threading
import time
import unittest
import pytest
from unittest.mock import patch

from sqlalchemy import create_engine, text
from sqlmemo.sqlmemo import SQLMemo, SQLMemoState


class TestCachedFunction:

    DATA_PATH = Path(__file__).parent / "data"

    def _is_hit(self, cached_f, *args, **kwargs):
        return cached_f._sqlmemo.get_record(*args, **kwargs) is not None

    def test_simple_caching(self):
        count = 0

        # Define a function to be cached
        @SQLMemo(store_default_args=True)
        def expensive_function(x, y=3, z=True):
            nonlocal count
            count += 1
            return x + y

        assert expensive_function(2, 3) == 5
        assert expensive_function(2, 3) == 5
        assert expensive_function(3, 3) == 6
        # Assert that the function was called only once
        assert count == 2
        # Parameter variations
        assert expensive_function(2) == 5
        assert expensive_function(2, y=3) == 5
        assert expensive_function(2, z=True) == 5
        assert count == 2

    def test_fib(self):
        count = 0

        @SQLMemo()
        def fib(x):
            nonlocal count
            count += 1
            return fib(x - 1) + fib(x - 2) if x > 1 else x

        assert fib(10) == 55
        assert count == 11

    def test_tables_and_funcnames(self):
        engine = create_engine("sqlite:///:memory:")
        count1 = 0
        count2 = 0
        count3 = 0

        @SQLMemo(engine, table_name="tab0", func_name="fun1")
        def fun1(x, y=3, z=True):
            nonlocal count1
            count1 += 1
            return x + y

        @SQLMemo(engine, table_name="tab0", func_name="fun1")
        def fun2(x, y=4, z=True):
            nonlocal count2
            count2 += 1
            return x + y + 1

        @SQLMemo(engine, table_name="tab1", func_name="fun1")
        def fun3(x, y=4, z=True):
            nonlocal count3
            count3 += 1
            return x + y + 2

        assert fun1(2, 3) == 5
        assert fun1(3, 3) == 6
        assert count1 == 2
        assert fun2(2, 3) == 5  # NB!!!
        assert count2 == 0
        assert fun2(2, 4) == 7
        assert count2 == 1
        assert fun3(2, 4) == 8
        assert count1 == 2
        assert count2 == 1
        assert count3 == 1

    # WIP below: JSON, Pickle, exceptions (store, rerun/reraise), parallel/running, ...
    # FIX: Json columns with None vs 'none'

    def test_caching_with_exception(self):
        @SQLMemo()
        def error_function():
            raise ValueError("Something went wrong")

        with pytest.raises(ValueError):
            error_function()
        with pytest.raises(ValueError):
            error_function()
        assert error_function._sqlmemo.stats.misses == 2

    def test_hash_args(self):
        """Tests the stability of the hashes."""
        cached_func = SQLMemo()

        @cached_func
        def my_function(a, b, c):
            return a + b + c

        assert cached_func._hash_args(1, 2, 3) == "08934e18c19ac2c1d1fd021bc6e33a522ef6fdadf9ffd082235ab9f09d02c519"
        assert cached_func._hash_args(1, 2, c=3) == "08934e18c19ac2c1d1fd021bc6e33a522ef6fdadf9ffd082235ab9f09d02c519"
        assert cached_func._hash_args(2, 3, 4) == "24242d2d2a29d91d628442c427c9890d8d9ce8a7d74b7fee56323521bec4c923"

    def test_get_record_by_hash(self):
        cached_func = SQLMemo()

        @cached_func
        def my_function(a, b, c):
            return a + b + c

        my_function(1, 2, 3)
        record = cached_func._get_record_by_hash("08934e18c19ac2c1d1fd021bc6e33a522ef6fdadf9ffd082235ab9f09d02c519")
        assert record and record.value_pickle
        assert pickle.loads(record.value_pickle) == 6
        assert record.state == SQLMemoState.DONE

    def test_get_record_by_args(self):
        cached_func = SQLMemo()

        @cached_func
        def my_function(a, b, c):
            return a + b + c

        my_function(b=2, a=1, c=3)
        record = cached_func.get_record(1, 2, c=3)
        assert record and record.value_pickle
        assert pickle.loads(record.value_pickle) == 6
        assert record.state == SQLMemoState.DONE

    def test_caching_counting_and_clearing(self):
        engine = create_engine("sqlite:///:memory:")

        @SQLMemo(engine)
        def f1(x):  # type: ignore
            return x + 1

        c1: SQLMemo = f1._sqlmemo

        @SQLMemo(engine)
        def f2(x):
            return x + 2

        c2: SQLMemo = f2._sqlmemo

        assert f1(1) == 2
        assert f1(2) == 3
        assert f1(2) == 3
        for _ in range(4):
            assert f2(1) == 3

        assert c1.stats.misses == 2
        assert c1.stats.hits == 1
        assert c1.get_db_stats().records == 2
        assert c2.stats.misses == 1
        assert c2.stats.hits == 3
        assert c2.get_db_stats().records == 1

        @SQLMemo(engine)
        def f1(x):
            return x + 1

        c1: SQLMemo = f1._sqlmemo

        assert f1(1) == 2
        assert f1(2) == 3
        assert f1(3) == 4
        assert f1(3) == 4
        assert c1.stats.misses == 1
        assert c1.stats.hits == 3
        assert c1.get_db_stats().records == 3

        c1.trim()
        assert c1.get_db_stats().records == 0
        assert f1(1) == 2
        assert f1(5) == 6
        assert f2(1) == 3
        # Note this includes the 1 from previous block
        assert c1.stats.misses == 3
        assert c1.stats.hits == 3
        assert c1.get_db_stats().records == 2
        assert c2.stats.misses == 1
        assert c2.stats.hits == 4
        assert c2.get_db_stats().records == 1

    def test_stats_and_trim(self):

        @SQLMemo()
        def f(x):
            time.sleep(0.01)
            if x == 42:
                raise ValueError("This is a test")
            return x + 1

        for i in range(10):
            f(i)
        with pytest.raises(ValueError):
            f(42)
        f(1)

        st = f._sqlmemo.get_db_stats()
        assert st.records == 10
        assert st.records_running == 0
        assert st.records_done == 10
        assert st.records_error == 0
        assert f._sqlmemo.stats.hits == 1
        assert f._sqlmemo.stats.misses == 11
        assert f._sqlmemo.stats.errors == 1

        f._sqlmemo.trim(15)
        st = f._sqlmemo.get_db_stats()
        assert st.records == 10

        f._sqlmemo.trim(5)
        st = f._sqlmemo.get_db_stats()
        assert st.records == 5
        assert not self._is_hit(f, 42)
        for i in range(10):
            assert self._is_hit(f, i) == (i >= 5)

        f._sqlmemo.trim()
        st = f._sqlmemo.get_db_stats()
        assert st.records == 0
        assert st.records_done == 0

    def test_parallel_calls(self, tmp_path):
        # @CachedFunction(f"sqlite:///{tmp_path}/tst.sqlite")
        @SQLMemo()
        def slow_function(x):
            time.sleep(0.1)
            return x * 2

        def run_function(x):
            return slow_function(x)

        # Test parallel calls with same argument
        threads = [threading.Thread(target=run_function, args=(0,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = slow_function._sqlmemo.get_db_stats()
        assert stats.records == 1

        # Test parallel calls with different arguments
        threads = [threading.Thread(target=run_function, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = slow_function._sqlmemo.get_db_stats()
        assert stats.records == 5

    def test_parallel_trim(self, tmp_path):
        # @CachedFunction(f"sqlite:///{tmp_path}/tst.sqlite")
        cache = SQLMemo()

        @cache
        def slow_function(x):
            time.sleep(0.01)
            if x == 1:
                cache.trim()
            time.sleep(0.05)
            return x * 2

        threads = [threading.Thread(target=slow_function, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cache.get_db_stats().records == 5

    def test_exception_reraising(self):
        @SQLMemo(record_exceptions=True, reraise_exceptions=True)
        def fn(x):
            if x == 0:
                raise ValueError("Cannot divide by zero")
            return 10 // x

        assert fn(2) == 5
        with pytest.raises(ValueError):
            fn(0)
        with pytest.raises(ValueError):
            fn(0)

        stats = fn._sqlmemo.get_db_stats()
        assert stats.records == 2
        assert stats.records_error == 1
        assert fn._sqlmemo.stats.errors == 1
        assert fn._sqlmemo.stats.hits == 1
        assert fn._sqlmemo.stats.misses == 2

    def test_json_pickle_storage(self):
        @SQLMemo(store_args_json=True, store_args_pickle=True, store_value_json=True)
        def fn(x, y):
            return {"result": x + y, "factors": [x, y]}

        result = fn(3, 4)
        assert result == {"result": 7, "factors": [3, 4]}
        cache_entry = fn._sqlmemo.get_record(3, 4)
        assert cache_entry is not None

        assert cache_entry.args_json == {"x": 3, "y": 4}
        assert cache_entry.value_json == result
        assert cache_entry.args_pickle is not None
        assert cache_entry.value_pickle is not None
        assert pickle.loads(cache_entry.args_pickle) == {"x": 3, "y": 4}
        assert pickle.loads(cache_entry.value_pickle) == result

    def test_dill_serialization(self):
        @SQLMemo(use_dill=True)
        def lambda_mult(x):
            if x > 0:
                return lambda a: a * x
            else:

                class Foo:
                    def __call__(self, a):
                        return a + x

                return Foo()

        assert lambda_mult(5)(2) == 10
        assert lambda_mult(5)(2) == 10
        assert lambda_mult(3)(4) == 12

        assert lambda_mult(-5)(10) == 5
        assert lambda_mult(-2)(-3) == -5
        assert lambda_mult(-5)(10) == 5

        assert lambda_mult._sqlmemo.stats.hits == 2
        assert lambda_mult._sqlmemo.stats.misses == 4
        assert lambda_mult._sqlmemo.get_db_stats().records == 4

    def test_read_db_file_plain(self, tmp_path):
        db_src = self.DATA_PATH / "test_data_plain.sqlite"
        db_path = tmp_path / "test_data_plain.sqlite"
        shutil.copy(db_src, db_path)

        @SQLMemo(db_path, reraise_exceptions=True, func_name="f", table_name="foo_bar")
        def f(x, *args, y=42, z=None, **kwargs):
            raise NotImplementedError("Should not be called")

        stats = f._sqlmemo.get_db_stats()
        assert stats.records_error == 1
        assert stats.records == 5

        assert f(1) == dict(x=1, args=(), kwargs={}, g=None)
        assert f(1, z="x", q=b"hello\0") == dict(x=1, args=(), kwargs=dict(q=b"hello\0"), g=None)
        assert f(1, 2, 3, foo="bar") == dict(x=1, args=(2, 3), kwargs=dict(foo="bar"), g=None)
        # More complicated comparison due to nan values
        r = f(1.3, float("nan"), float("inf"), z=frozenset({1, 2, 4}), w=SQLMemoState.ERROR)
        assert math.isnan(r["args"][0])
        assert math.isinf(r["args"][1])
        del r["args"]
        assert r == dict(x=1.3, kwargs=dict(w=SQLMemoState.ERROR), g={1, 2, 3, 4})

        with pytest.raises(NotImplementedError):
            f("absent")
        with pytest.raises(ValueError):
            f("err")

    def test_read_db_file_dill(self, tmp_path):
        db_src = self.DATA_PATH / "test_data_dill.sqlite"
        db_path = tmp_path / "test_data_dill.sqlite"
        shutil.copy(db_src, db_path)

        @SQLMemo(db_path, reraise_exceptions=True, func_name="f", use_dill=True)
        def f(x, *args, y=42, z=None, **kwargs):
            raise NotImplementedError("Should not be called")

        stats = f._sqlmemo.get_db_stats()
        assert stats.records_error == 1
        assert stats.records == 5

        assert f(1) == dict(x=1, args=(), kwargs={}, g=b"hello\1")
        result = f(1, z="x", q=b"hello\0")
        assert result["x"] == 1
        assert result["args"] == ()
        assert result["kwargs"] == {"z": "x", "q": b"hello\0"}
        assert callable(result["g"])  # lambda function
        assert result["g"](3) == 6

        result = f(1, 2, 3, foo="bar")
        assert result["x"] == 1
        assert result["args"] == (2, 3)
        assert result["kwargs"] == {"foo": "bar"}
        assert hasattr(result["g"], "foo")
        assert result["g"].foo() == 3

        result = f(1.3, float("nan"), float("inf"), z=frozenset({1, 2, 4}), w=SQLMemoState.ERROR)
        assert result["x"] == 1.3
        assert math.isnan(result["args"][0])
        assert math.isinf(result["args"][1])
        assert result["kwargs"] == {"z": frozenset({1, 2, 4}), "w": SQLMemoState.ERROR}
        assert callable(result["g"])
        assert result["g"]() == 6  # Foo(2).foo()

        with pytest.raises(ValueError):
            f("err")

    def test_read_db_file_json(self, tmp_path):
        db_src = self.DATA_PATH / "test_data_json.sqlite"
        db_path = tmp_path / "test_data_json.sqlite"
        shutil.copy(db_src, db_path)

        @SQLMemo(db_path, reraise_exceptions=True, func_name="f", store_args_json=True, store_value_json=True)
        def f(x, *args, y=42, z=None, **kwargs):
            raise NotImplementedError("Should not be called")

        stats = f._sqlmemo.get_db_stats()
        assert stats.records_error == 1
        assert stats.records == 4

        assert f(1) == dict(x=1, args=(), kwargs={})
        assert f(1, z="x", q=(1, 2, 3)) == dict(x=1, args=(), kwargs={"z": "x", "q": (1, 2, 3)})
        assert f(1, 2, 3, foo="bar") == dict(x=1, args=(2, 3), kwargs={"foo": "bar"})

        with pytest.raises(ValueError):
            f("err")


# TODO: Add tests for argument filtering (json, pickle), value filtering (json), and exception filtering (in recording and reraising)

if __name__ == "__main__":
    unittest.main()
