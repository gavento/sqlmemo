import dataclasses
import hashlib
from typing import Any


def _hash_helper(obj: Any, hash, sort_keys: bool):
    hash.update(str(type(obj)).encode())
    if isinstance(obj, (int, float, bool, str, bytes, type(None))):
        hash.update(str(obj).encode())  # These have a deterministic str()
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _hash_helper(item, hash, sort_keys=sort_keys)
    elif isinstance(obj, (set, frozenset)):
        for item in sorted(obj) if sort_keys else obj:
            _hash_helper(item, hash, sort_keys=sort_keys)
    elif isinstance(obj, dict):
        # Note: the order of items in the dictionary generaly matters
        keys = sorted(obj.keys()) if sort_keys else obj.keys()
        for key in keys:
            _hash_helper(key, hash, sort_keys=sort_keys)
            _hash_helper(obj[key], hash, sort_keys=sort_keys)
    elif dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            _hash_helper(field.name, hash, sort_keys=sort_keys)
            _hash_helper(getattr(obj, field.name), hash, sort_keys=sort_keys)
    else:
        raise TypeError(f"Unsupported type in hash_obj: {type(obj)}")


def hash_obj(obj: Any, sort_keys: bool = True, hash_factory=hashlib.sha256) -> str:
    """
    Smart object hasher that can work with dataclasses, iterables, float NaNs, etc.

    Also hashes the types of the parameters (so e.g. `(1,2)` and `[1,2]` are distinct).
    By default, dictionaries are hashed including the ordering of the values,
    set `sort_dicts=True` to have their keys sorted. Raises `TypeError` on an unknown type.
    Uses SHA256 from hashlib by default.
    """
    hash = hash_factory()
    _hash_helper(obj, hash, sort_keys=sort_keys)
    return hash.hexdigest()


def jsonize(obj: Any) -> str:
    """Converts an object to a JSON-compatible object, with dataclasses converted to dicts.
    
    """
    if isinstance(obj, (int, bool, str, float, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [jsonize(item) for item in obj]
    elif isinstance(obj, (set, frozenset)):
        return [jsonize(item) for item in obj] #NB: The ordering is arbitrary
    elif isinstance(obj, dict):
        for k in obj.keys():
            if not isinstance(k, str):
                raise TypeError(
                    f"Dictionary keys must be strings to be JSON-compatible, found: {type(k)}"
                )
        return {k: jsonize(v) for k, v in obj.items()}
    elif dataclasses.is_dataclass(obj):
        return {
            field.name: jsonize(getattr(obj, field.name))
            for field in dataclasses.fields(obj)
        }
    else:
        raise TypeError(f"Unsupported type in jsonize: {type(obj)}")

