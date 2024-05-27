import inspect
import types
import typing
from dataclasses import fields, is_dataclass
from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    PickleType,
    String,
    Table,
)
from sqlalchemy import func as sql_func

# Define common type mapping
TYPE_MAP = {
    int: Integer,
    str: String,
    float: Float,
    datetime: DateTime,
    bool: Integer,  # SQLAlchemy does not have a Boolean type for all databases
}


def _in_list_or_is_true(value, lst: list | bool) -> bool:
    """Return True if value is in lst, or lst is True."""
    return value in lst if isinstance(lst, list) else lst


def _filter_subtree(items: list[str] | bool, prefix: str) -> list[str] | bool:
    """Return a list of strings that start with prefix followed by "." or end, removing that prefix. If items is bool, return it as is."""
    if isinstance(items, bool):
        return items
    return [
        s[len(prefix) :] for s in items if s == prefix or s.startswith(f"{prefix}.")
    ]


def type_to_columns(
    prefix: str,
    type_,
    default_=None,  # Only applies to this column, not recursively
    nullable_=False,  # Only applies to this column, not recursively
    expand: (
        list[str] | bool
    ) = True,  # Bool or list of paths, each startng with ".", or an empty string for "this"
    pickle: (
        list[str] | bool
    ) = False,  # Bool or list of paths, each startng with ".", or an empty string for "this"
    json: (
        list[str] | bool
    ) = False,  # Bool or list of paths, each startng with ".", or an empty string for "this"
) -> list[Column]:
    # If default_ is empty param from inspect, set it to None
    if default_ is inspect.Parameter.empty:
        default_ = None
    # Special: if type_ is a tuple, it is treated as a tuple of types
    if isinstance(type_, tuple):
        type_ = tuple[type_]

    # Logic for deciding what to do: expand, pickle, or json
    # Priority: if "" is present in any of the lists (and they are lists), it takes precedence
    # If not, then the boolean values are used
    # If these conflict ("" present in more than one list, or pickle and json are set), an error is raised
    do_expand, do_pickle, do_json = False, False, False
    if isinstance(expand, list) and "" in expand:
        do_expand = True
        assert not isinstance(pickle, list) or "" not in pickle
        assert not isinstance(json, list) or "" not in json
    elif isinstance(pickle, list) and "" in pickle:
        do_pickle = True
        assert not isinstance(json, list) or "" not in json
    elif isinstance(json, list) and "" in json:
        do_json = True
    elif expand is True:
        do_expand = True
        # Expand=True takes precedence over pickle=True and json=True for dataclasses and tuples
    elif pickle is True:
        do_pickle = True
        assert not (json is True)
    elif json is True:
        do_json = True

    # Check if type is a Union of a basic type and None, using typing.get_origin
    if typing.get_origin(type_) == types.UnionType and type(None) in typing.get_args(
        type_
    ):
        assert (
            len(typing.get_args(type_)) == 2
        ), f"Union with None should have only two arguments, found {type_}"
        # If the type is a Union of a basic type and None, use the basic type
        type_ = next(t for t in typing.get_args(type_) if t is not type(None))
        nullable_ = True

    # Construct the list of columns
    columns = []

    # Check if the type is a basic type
    if type_ in TYPE_MAP:
        col_type = TYPE_MAP[type_]
        # check if there is a default value provided for the parameter, and if so set it as the default val for the column
        columns.append(
            Column(f"{prefix}", col_type, nullable=nullable_, default=default_)
        )

    # Check if the type is a dataclass AND should be expanded AND was not explicitely declared to be pickled/JSONed
    elif is_dataclass(type_) and do_expand:
        assert (
            not nullable_
        ), "Expanded dataclasses cannot be nullable (library limitation)"
        for field in fields(type_):
            columns.extend(
                type_to_columns(
                    f"{prefix}.{field.name}",
                    field.type,
                    default_=(
                        field.default
                        if field.default is not inspect.Parameter.empty
                        else None
                    ),
                    expand=_filter_subtree(expand, f".{field.name}"),
                    pickle=_filter_subtree(pickle, f".{field.name}"),
                    json=_filter_subtree(json, f".{field.name}"),
                )
            )

    # Check if the type is a tuple AND should be expanded AND was not explicitely declared to be pickled/JSONed
    elif isinstance(type_, tuple) and do_expand:
        for i, subtype in enumerate(type_):
            columns.extend(
                type_to_columns(
                    f"{prefix}.{i}",
                    subtype,
                    expand=_filter_subtree(expand, f".{i}"),
                    pickle=_filter_subtree(pickle, f".{i}"),
                    json=_filter_subtree(json, f".{i}"),
                )
            )

    # Check if the field is to be stored as JSON
    elif is_dataclass(type_) and do_json:
        columns.append(Column(f"{prefix}", JSON, nullable=nullable_))
    # Check if the field is to be stored as a pickle
    elif do_pickle:
        columns.append(Column(f"{prefix}", PickleType, nullable=nullable_))
    else:
        raise TypeError(f"Cannot convert type {type_} to a column ({prefix})")
    return columns


def create_table_for_function(
    func,
    table_name=None,
    *,
    expand_arg: list[str] | bool = True,
    expand_ret: list[str] | bool = True,
    json_arg: list[str] | bool = False,
    json_ret: list[str] | bool = False,
    pickle_arg: list[str] | bool = False,  # This is unsupported and should not be used
    pickle_ret: list[str] | bool = True,
) -> Table:
    """Create a SQLAlchemy Table for the given function."""
    assert pickle_arg is None, "Pickle for arguments is unstable when used as key"
    # Extract the function's signature
    func_signature = inspect.signature(func)
    metadata = MetaData()

    # Define the columns for the table
    columns = [
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("timestamp", DateTime, default=sql_func.now()),
        Column("host", String),
        Column("user", String),
        Column("status", String),
        Column("error", PickleType, nullable=True),
        Column("arg_args", JSON, nullable=True),
        Column("arg_kwargs", JSON, nullable=True),
    ]

    # Add columns for each argument in the function
    for arg_name, arg_param in func_signature.parameters.items():
        arg_type = arg_param.annotation
        nullable = arg_param.default is not inspect.Parameter.empty
        columns.extend(
            type_to_columns(
                f"arg.{arg_name}",
                arg_type,
                nullable=nullable,
                expand=expand_arg,
                pickle=pickle_arg,
                json=json_arg,
            )
        )

    # Add a single column, or more columns, for the return value
    return_annotation = func_signature.return_annotation
    columns.extend(
        type_to_columns(
            "ret",
            return_annotation,
            expand=expand_ret,
            pickle=pickle_ret,
            json=json_ret,
        )
    )

    # Set the table name if not provided
    if table_name is None:
        table_name = f"{func.__name__}_cache"

    # Define and return the table
    return Table(table_name, metadata, *columns, extend_existing=True)
