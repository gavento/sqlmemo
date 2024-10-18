import pickle
import time
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine, text
from sqlcache.cached_function import CachedFunction, FunctionState


class CachedFunctionTest(unittest.TestCase):

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

        assert fun1(2, 3) == 5
        assert fun1(3, 3) == 6
        assert count1 == 2
        assert fun2(2, 3) == 5  # NB!!!
        assert count2 == 0
        assert fun2(2, 4) == 7
        assert count2 == 1

    # WIP below: JSON, Pickle, exceptions (store, rerun/reraise), parallel/running, ...
    # FIX: Json columns with None vs 'none'

    def test_caching_with_exception(self):
        @CachedFunction()
        def error_function():
            raise ValueError("Something went wrong")

        with self.assertRaises(ValueError):
            error_function()
        with self.assertRaises(ValueError):
            error_function()
        self.assertEqual(error_function._sqlcache.get_stats().misses, 2)

    def test_hash_args(self):
        """Tests the stability of the hashes."""
        cached_func = CachedFunction()

        @cached_func
        def my_function(a, b, c):
            return a + b + c

        self.assertEqual(cached_func.hash_args(
            1, 2, 3), "08934e18c19ac2c1d1fd021bc6e33a522ef6fdadf9ffd082235ab9f09d02c519")
        self.assertEqual(cached_func.hash_args(
            1, 2, c=3), "08934e18c19ac2c1d1fd021bc6e33a522ef6fdadf9ffd082235ab9f09d02c519")
        self.assertEqual(cached_func.hash_args(
            2, 3, 4), "24242d2d2a29d91d628442c427c9890d8d9ce8a7d74b7fee56323521bec4c923")

    def test_get_record_by_hash(self):
        cached_func = CachedFunction()

        @cached_func
        def my_function(a, b, c):
            return a + b + c

        my_function(1, 2, 3)
        record = cached_func.get_record_by_hash(
            "08934e18c19ac2c1d1fd021bc6e33a522ef6fdadf9ffd082235ab9f09d02c519")
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
        with self.assertRaises(ValueError):
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


if __name__ == "__main__":
    unittest.main()
