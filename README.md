

# SQL-backed computation cache


## Parallelism



## Async

WIP

## What counts as the same invocation?

The cache is indexed by:

- The table name derived from qualified function name (`func.__qualname__`) or given as `table_name`. If you want more functions to share a table, just set the same tablename.
- The qualified function name (`func.__qualname__`) or given as `func_name`. If you rename or move a function and want to keep using the cache, set `func_name` and `table_name` explicitly.
- SHA256 hash of all the arguments. This means the following:
  - We recursively walk the argument structure, not all types are supported. Dictionary and set keys are sorted (i.e. ordered dictionary ordering is ignored). `__hash__` is ignored because it is not stable between runs. 
  - The default argument values are applied (so for `def f(x=42)`, `f()` and `f(42)` are identical).
  - It does not matter if *declared* arguments are given by name or position (e.g. `f(42)` and `f(x=42)` are the same)
  - If your function has `*args` and/or `**kwargs` parameters, these are stored just as `args` resp `kwargs` parameters (or named however you named them).
  - The names of the parameters matter - if you rename a parameter, the calls will have different hashes.
  - If you need to know more details, the actual code is equivalent to: `b = inspect.signature(func).bind(*args, **kwargs); b.apply_defaults(); return b.arguments`


