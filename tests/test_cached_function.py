import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlcache.cached_function import CachedFunction, FunctionState

class CachedFunctionTest(unittest.TestCase):

    def test_caching(self):
        count = 0

        # Define a function to be cached
        @CachedFunction()
        def expensive_function(x, y):
            nonlocal count
            count += 1
            return x + y

        result1 = expensive_function(2, 3)
        result2 = expensive_function(2, 3)
        # Assert that the results are equal
        assert result1 == result2
        # Assert that the function was called only once
        assert count == 1

    def test_caching_with_exception(self):
        # Create a CachedFunction instance
        cached_func = CachedFunction()

        # Define a function that raises an exception
        @cached_func
        def error_function():
            raise ValueError("Something went wrong")

        # Call the function and catch the exception
        with self.assertRaises(ValueError):
            error_function()

        # Call the function again
        result = error_function()

        # Assert that the result is None
        self.assertIsNone(result)

        # Assert that the function was called twice
        self.assertEqual(error_function._func.call_count, 2)

    def test_caching_with_running_state(self):
        # Create a CachedFunction instance
        cached_func = CachedFunction()

        # Define a function that takes a long time to execute
        @cached_func
        def long_running_function():
            import time
            time.sleep(5)
            return "Done"

        # Call the function in a separate thread
        with patch("sqlcache.cached_function.threading.Thread") as mock_thread:
            mock_thread.return_value.is_alive.return_value = True
            result = long_running_function()

        # Assert that the result is None
        self.assertIsNone(result)

        # Assert that the function was called once
        self.assertEqual(long_running_function._func.call_count, 1)

        # Assert that the function state is set to "RUNNING"
        self.assertEqual(long_running_function._table.insert.call_args[1]["state"], FunctionState.RUNNING.value)


class TestCache(unittest.TestCase):

    def test_basic_inc(self):
        engine = create_engine("sqlite:///:memory:")

        inc_used = 0

        @CachedFunction(engine, table="inc")
        def inc(x: int):
            nonlocal inc_used
            inc_used += 1
            return x + 1

        assert inc(1) == 2
        assert inc(1) == 2
        assert inc_used == 1
        assert inc(2) == 3

        inc2_used = 0

        @CachedFunction(engine, table="inc")
        def inc2(x: int):
            nonlocal inc2_used
            inc2_used += 1
            return x + 1

        assert inc2(1) == 1
        assert inc2(1) == 1
        assert inc2_used == 0
        assert inc2(3) == 4


    def test_rec_and_async(self):
        engine = create_engine("sqlite:///:memory:")

        fib_used = 0

        @CachedFunction(engine)
        def fib(x):
            nonlocal fib_used
            fib_used += 1
            return fib(x - 1) + fib(x - 2) if x > 1 else x

        assert fib(10) == 55
        assert fib_used == 11

        afib_used = 0

        @CachedFunction(engine)
        async def afib(x):
            nonlocal afib_used
            afib_used += 1
            return await afib(x - 1) + await afib(x - 2) if x > 1 else x

        # Run in an executor to avoid blocking the event loop

        assert asyncio.run(afib(10)) == 55
        assert afib_used == 11


if __name__ == "__main__":
    unittest.main()