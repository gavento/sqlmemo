import asyncio
from functools import wraps

from sqlalchemy import Engine


def cached(engine: Engine):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if the function is async
            if asyncio.iscoroutinefunction(func):

                async def async_wrapper():
                    # Your caching logic goes here
                    # Use the SQLAlchemy engine to cache the results
                    # ...
                    return await func(*args, **kwargs)

                return async_wrapper()
            else:

                def sync_wrapper():
                    # Your caching logic goes here
                    # Use the SQLAlchemy engine to cache the results
                    # ...
                    return func(*args, **kwargs)

                return sync_wrapper()

        return wrapper

    return decorator
