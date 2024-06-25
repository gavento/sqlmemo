import pickle
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine, text
from sqlcache.cached_function import CachedFunction, FunctionState


class CachedFunctionTest(unittest.TestCase):

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

        # with fun1.__sqlcache__._get_locked_session() as session:
        #     print(session.execute(text("SELECT * FROM tab0")).fetchall())

        assert fun2(2, 3) == 5 # NB!!!
        assert count2 == 0
        assert fun2(2, 4) == 7
        assert count2 == 1

    # WIP below: JSON, Pickle, exceptions (store, rerun/reraise), parallel/running, ...
    # FIX: Json columns with None vs 'none'

    def test_caching_with_exception(self):
        cached_func = CachedFunction()

        @cached_func
        def error_function():
            raise ValueError("Something went wrong")

        with self.assertRaises(ValueError):
            error_function()
        with self.assertRaises(ValueError):
            error_function()
        self.assertEqual(error_function._func.call_count, 2)

    def test_hash_args(self):
        cached_func = CachedFunction()

        @cached_func
        def my_function(a, b, c):
            return a + b + c

        self.assertEqual(cached_func.hash_args(1, 2, 3), "08934e18c19ac2c1d1fd021bc6e33a522ef6fdadf9ffd082235ab9f09d02c519")
        self.assertEqual(cached_func.hash_args(1, 2, c=3), "08934e18c19ac2c1d1fd021bc6e33a522ef6fdadf9ffd082235ab9f09d02c519")
        self.assertEqual(cached_func.hash_args(2, 3, 4), "24242d2d2a29d91d628442c427c9890d8d9ce8a7d74b7fee56323521bec4c923")

    def test_get_record_by_hash(self):
        cached_func = CachedFunction()

        @cached_func
        def my_function(a, b, c):
            return a + b + c

        my_function(1, 2, 3)
        record = cached_func.get_record_by_hash("08934e18c19ac2c1d1fd021bc6e33a522ef6fdadf9ffd082235ab9f09d02c519")
        assert pickle.loads(record["return_pickle"]) == 6
        assert record["state"] == FunctionState.DONE.value

    def test_get_record_by_args(self):
        cached_func = CachedFunction()

        @cached_func
        def my_function(a, b, c):
            return a + b + c

        my_function(b=2, a=1, c=3)
        record = cached_func.get_record_by_args(1, 2, c=3)
        assert pickle.loads(record["return_pickle"]) == 6
        assert record["state"] == FunctionState.DONE.value

if __name__ == "__main__":
    unittest.main()
