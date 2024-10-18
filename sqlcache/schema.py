import datetime
import enum
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    Index,
    LargeBinary,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped


class FunctionState(enum.Enum):
    RUNNING = "RUNNING"
    DONE = "DONE"
    ERROR = "ERROR"


class Base(DeclarativeBase):
    type_annotation_map = {
        datetime.datetime: TIMESTAMP(timezone=True),
        bytes: LargeBinary,
    }


class CachedFunctionEntry(Base):
    __tablename__ = 'cached_function'

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[float] # Unix UTC timestamp
    func_name: Mapped[str]
    args_hash: Mapped[str]
    state: Mapped[FunctionState]
    user: Mapped[Optional[str]]
    hostname: Mapped[Optional[str]]
    runtime_seconds: Mapped[Optional[float]]
    args_pickle: Mapped[Optional[bytes]]
    args_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    return_pickle: Mapped[Optional[bytes]]
    return_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    exception_pickle: Mapped[Optional[bytes]]
    exception_str: Mapped[Optional[Any]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index('ix_cached_function_func_name_args_hash',
              'func_name', 'args_hash', unique=True),
        Index('ix_cached_function_func_name_state_timestamp',
              'func_name', 'state', 'timestamp'),
    )

