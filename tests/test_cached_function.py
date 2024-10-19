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
from sqlcache.cached_function import CachedFunction, FunctionState


class TestCachedFunction:

    DATA_PATH = Path(__file__).parent / "data"

    def _is_hit(self, cached_f, *args, **kwargs):
        return cached_f._sqlcache.get_record(*args, **kwargs) is not None

    def test_simple_caching(self):
        count = 0

        # Define a function to be cached
        @CachedFunction()
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

        @CachedFunction()
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

        @CachedFunction(engine, table_name="tab0", func_name="fun1")
        def fun1(x, y=3, z=True):
            nonlocal count1
            count1 += 1
            return x + y

        @CachedFunction(engine, table_name="tab0", func_name="fun1")
        def fun2(x, y=4, z=True):
            nonlocal count2
            count2 += 1
            return x + y + 1

        @CachedFunction(engine, table_name="tab1", func_name="fun1")
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
        @CachedFunction()
        def error_function():
            raise ValueError("Something went wrong")

        with pytest.raises(ValueError):
            error_function()
        with pytest.raises(ValueError):
            error_function()
        assert error_function._sqlcache.get_stats().misses == 2

    def test_hash_args(self):
        """Tests the stability of the hashes."""
        cached_func = CachedFunction()

        @cached_func
        def my_function(a, b, c):
            return a + b + c

        assert cached_func.hash_args(1, 2, 3) == "08934e18c19ac2c1d1fd021bc6e33a522ef6fdadf9ffd082235ab9f09d02c519"
        assert cached_func.hash_args(1, 2, c=3) == "08934e18c19ac2c1d1fd021bc6e33a522ef6fdadf9ffd082235ab9f09d02c519"
        assert cached_func.hash_args(2, 3, 4) == "24242d2d2a29d91d628442c427c9890d8d9ce8a7d74b7fee56323521bec4c923"

    def test_get_record_by_hash(self):
        cached_func = CachedFunction()

        @cached_func
        def my_function(a, b, c):
            return a + b + c

        my_function(1, 2, 3)
        record = cached_func.get_record_by_hash("08934e18c19ac2c1d1fd021bc6e33a522ef6fdadf9ffd082235ab9f09d02c519")
        assert record and record.value_pickle
        assert pickle.loads(record.value_pickle) == 6
        assert record.state == FunctionState.DONE

    def test_get_record_by_args(self):
        cached_func = CachedFunction()

        @cached_func
        def my_function(a, b, c):
            return a + b + c

        my_function(b=2, a=1, c=3)
        record = cached_func.get_record(1, 2, c=3)
        assert record and record.value_pickle
        assert pickle.loads(record.value_pickle) == 6
        assert record.state == FunctionState.DONE

    def test_caching_counting_and_clearing(self):
        engine = create_engine("sqlite:///:memory:")

        @CachedFunction(engine)
        def f1(x):  # type: ignore
            return x + 1

        c1: CachedFunction = f1._sqlcache

        @CachedFunction(engine)
        def f2(x):
            return x + 2

        c2: CachedFunction = f2._sqlcache

        assert f1(1) == 2
        assert f1(2) == 3
        assert f1(2) == 3
        for _ in range(4):
            assert f2(1) == 3

        assert c1.get_stats().misses == 2
        assert c1.get_stats().hits == 1
        assert c1.get_stats().cache_size == 2
        assert c2.get_stats().misses == 1
        assert c2.get_stats().hits == 3
        assert c2.get_stats().cache_size == 1

        @CachedFunction(engine)
        def f1(x):
            return x + 1

        c1: CachedFunction = f1._sqlcache

        assert f1(1) == 2
        assert f1(2) == 3
        assert f1(3) == 4
        assert f1(3) == 4
        assert c1.get_stats().misses == 1
        assert c1.get_stats().hits == 3
        assert c1.get_stats().cache_size == 3

        c1.trim_cache()
        assert c1.get_stats().cache_size == 0
        assert f1(1) == 2
        assert f1(5) == 6
        assert f2(1) == 3
        # Note this includes the 1 from previous block
        assert c1.get_stats().misses == 3
        assert c1.get_stats().hits == 3
        assert c1.get_stats().cache_size == 2
        assert c2.get_stats().misses == 1
        assert c2.get_stats().hits == 4
        assert c2.get_stats().cache_size == 1

    def test_stats_and_trim(self):

        @CachedFunction()
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

        st = f._sqlcache.get_stats()
        assert st.cache_size == 10
        assert st.cache_running == 0
        assert st.cache_done == 10
        assert st.cache_error == 0
        assert st.hits == 1
        assert st.misses == 11

        f._sqlcache.trim_cache(15)
        st = f._sqlcache.get_stats()
        assert st.cache_size == 10

        f._sqlcache.trim_cache(5)
        st = f._sqlcache.get_stats()
        assert st.cache_size == 5
        assert not self._is_hit(f, 42)
        for i in range(10):
            assert self._is_hit(f, i) == (i >= 5)

        f._sqlcache.trim_cache()
        st = f._sqlcache.get_stats()
        assert st.cache_size == 0
        assert st.cache_done == 0

    def test_parallel_calls(self, tmp_path):
        # @CachedFunction(f"sqlite:///{tmp_path}/tst.sqlite")
        @CachedFunction()
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

        stats = slow_function._sqlcache.get_stats()
        assert stats.cache_size == 1

        # Test parallel calls with different arguments
        threads = [threading.Thread(target=run_function, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = slow_function._sqlcache.get_stats()
        assert stats.cache_size == 5

    def test_parallel_trim(self, tmp_path):
        # @CachedFunction(f"sqlite:///{tmp_path}/tst.sqlite")
        cache = CachedFunction()

        @cache
        def slow_function(x):
            time.sleep(0.01)
            if x == 1:
                cache.trim_cache()
            time.sleep(0.05)
            return x * 2

        threads = [threading.Thread(target=slow_function, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cache.get_stats().cache_size == 5

    def test_exception_reraising(self):
        @CachedFunction(record_exceptions=True, reraise_exceptions=True)
        def error_function(x):
            if x == 0:
                raise ValueError("Cannot divide by zero")
            return 10 / x

        try:
            error_function(0)
        except ValueError:
            pass  # Expected exception

        stats = error_function._sqlcache.get_stats()
        assert stats.misses == 1
        assert stats.cache_error == 1

        try:
            error_function(0)
        except ValueError:
            pass  # Expected exception

        stats = error_function._sqlcache.get_stats()
        assert stats.misses == 1
        assert stats.hits == 1
        assert stats.cache_error == 1

    def test_json_pickle_storage(self):
        @CachedFunction(args_json=True, args_pickle=True, value_json=True)
        def complex_function(x, y):
            return {"result": x + y, "factors": [x, y]}

        result = complex_function(3, 4)
        assert result == {"result": 7, "factors": [3, 4]}
        cache_entry = complex_function._sqlcache.get_record(3, 4)

        assert cache_entry.args_json == {"x": 3, "y": 4}
        assert cache_entry.value_json == result
        assert pickle.loads(cache_entry.args_pickle) == {"x": 3, "y": 4}
        assert pickle.loads(cache_entry.value_pickle) == result

    def test_dill_serialization(self):
        @CachedFunction(use_dill=True)
        def lambda_function(x):
            return (lambda a: a * 2)(x)

        result = lambda_function(5)
        assert result == 10

        cache_entry = lambda_function._sqlcache.get_record(5)
        assert cache_entry is not None
        assert pickle.loads(cache_entry.value_pickle) == 10


if __name__ == "__main__":
    unittest.main()
