from sqlalchemy import Engine, create_engine

from .schema import create_table

DEFAULT_ENGINE = "sqlite:///:memory:"


class CachedFunction:
    def __init__(
        self, func: function, engine: Engine | str | function = DEFAULT_ENGINE
    ):
        # If engine is a callable, call it to get the engine
        if callable(engine):
            engine = engine()
        # If engine is a string, create an engine from it
        if isinstance(engine, str):
            engine = create_engine(engine)

        self.engine = engine
        self.func = func
        self.table = create_table(engine, func)
