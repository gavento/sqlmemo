import enum
from typing import Any, Optional, Type, TypeVar

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    Index,
    LargeBinary,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, declarative_base


class RecordState(enum.Enum):
    RUNNING = "RUNNING"
    DONE = "DONE"
    ERROR = "ERROR"


class MemoizeRecord:
    __abstract__ = True

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[float]  # Unix UTC timestamp
    func_name: Mapped[str]
    args_hash: Mapped[str]
    state: Mapped[RecordState]
    user: Mapped[Optional[str]]
    hostname: Mapped[Optional[str]]
    runtime_seconds: Mapped[Optional[float]]
    args_pickle: Mapped[Optional[bytes]]
    args_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    value_pickle: Mapped[Optional[bytes]]
    value_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    exception_pickle: Mapped[Optional[bytes]]
    exception_str: Mapped[Optional[Any]] = mapped_column(Text, nullable=True)


def concrete_memoize_record(table_name) -> Type[MemoizeRecord]:

    ## NB: We want to have separate metadata for each table, so we can't use the default DeclarativeBase
    class Base(DeclarativeBase):
        type_annotation_map = {
            bytes: LargeBinary,
        }

    class ConcreteMemoizeRecord(Base, MemoizeRecord):
        __abstract__ = False
        __tablename__ = table_name
        __table_args__ = (
            Index(f"ix__{table_name}__func_name__args_hash", "func_name", "args_hash", unique=True),
            Index(f"ix__{table_name}__func_name__state__timestamp", "func_name", "state", "timestamp"),
        )

    return ConcreteMemoizeRecord
