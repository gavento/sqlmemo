from sqlcache import cached
from sqlalchemy import create_engine

def test_basic_inc():
    engine = create_engine('sqlite:///:memory:')

    inc_used = 0
    @cached(engine, table='inc')
    def inc(x: int):
        nonlocal inc_used
        inc_used += 1
        return x + 1
    
    assert inc(1) == 2
    assert inc(1) == 2
    assert inc_used == 1
    assert inc(2) == 3

    inc2_used = 0
    @cached(engine, table='inc')
    def inc2(x : int):
        nonlocal inc2_used
        inc2_used += 1
        return x + 1
    
    assert inc2(1) == 1
    assert inc2(1) == 1
    assert inc2_used == 0
    assert inc2(3) == 4

def test_rec_and_async():
    engine = create_engine('sqlite:///:memory:')

    fib_used = 0
    @cached(engine)
    def fib(x):
        nonlocal fib_used
        fib_used += 1
        return fib(x - 1) + fib(x - 2) if x > 1 else x
    
    assert fib(10) == 55
    assert fib_used == 11

    afib_used = 0
    @cached(engine)
    async def afib(x):
        nonlocal afib_used
        afib_used += 1
        return await afib(x - 1) + await afib(x - 2) if x > 1 else x

    # Run in an executor to avoid blocking the event loop
    import asyncio
    assert asyncio.run(afib(10)) == 55
    assert afib_used == 11
