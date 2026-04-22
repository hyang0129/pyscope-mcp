# pycg Limitations

pycg is a **static** call-graph generator: it parses Python source into an AST and infers call edges without executing the code. Every edge it produces has to be derivable from syntax and scope alone. Anything whose target is only decided at runtime is invisible to it.

This document catalogues the patterns that defeat pycg so agents and humans using pycg-mcp know how to interpret its output.

## Mental model

Treat the graph as:

> **"Definite edges, with silent false negatives."**

- If pycg says A calls B, it almost certainly does.
- If pycg shows no edge between A and B, that's **weak** evidence they're unrelated — pycg may simply not have been able to resolve the target.

Use it to answer "who definitely depends on X" — not "is X dead code."

## Patterns pycg cannot resolve

### 1. Dynamic dispatch through a table

```python
HANDLERS = {"tts": run_tts, "llm": run_llm}
HANDLERS[task_type](payload)
```

pycg sees a dict lookup and a call on the result. It does not know which entry was retrieved, so no edge is drawn to `run_tts` or `run_llm` from this call site.

### 2. Strings-as-code

```python
mod = importlib.import_module(cfg["module"])
cls = getattr(mod, name)
cls()
```

The target is a string only known at runtime. pycg cannot follow `importlib.import_module`, `__import__`, or `getattr(obj, name)` with a non-literal name.

### 3. Decorator-based registries

```python
@registry.register("persona_a")
def run_a(ctx): ...

registry.dispatch("persona_a", ctx)
```

The registration stores `run_a` in a structure pycg does not track symbolically, and the dispatch call retrieves it by string key. Very common in agent, plugin, and router frameworks.

### 4. LLM tool-use

```python
tools = [search_web, render_frame, write_file]
client.messages.create(tools=tools, ...)
```

pycg sees the list being built, but the decision of which tool to invoke happens inside the model response. There is no static edge from the calling code to the tool implementations.

### 5. Duck-typed polymorphism

```python
def run_stage(stage):
    stage.execute(ctx)
```

If five classes implement `execute` and `stage` could be any of them, pycg may resolve to one, to none, or to a spurious target depending on how type information flows through the specific call site. Do not rely on the resolved target.

### 6. Conditional and lazy imports

```python
def encode(frame, use_gpu):
    if use_gpu:
        from .gpu import encode_impl
    else:
        from .cpu import encode_impl
    return encode_impl(frame)
```

Module-level conditional imports are usually captured. Imports inside functions are often missed, especially when gated on runtime values.

### 7. Metaclasses and `__init_subclass__`

Class hierarchies built by metaclasses, or side effects triggered in `__init_subclass__`, are opaque to pycg. Any call graph that depends on subclass registration will be incomplete.

### 8. `exec`, `eval`, templated code

```python
exec(compile(source, "<str>", "exec"), namespace)
```

Zero visibility. Treat any module that runs generated code as a black box in the graph.

### 9. C extensions and stdlib internals

Functions implemented in C (numpy operations, most of `builtins`, parts of `stdlib`) have no Python source to analyze. Calls into them terminate the trace. You may see edges like `<builtin>.dict.get` as placeholders.

### 10. Complex decorator stacks

Decorators that return wrappers built from `functools.wraps` usually resolve correctly. Decorators that replace the function with an instance of a callable class, or that compose multiple layers dynamically, can cause pycg to either mis-attribute the edge or to raise.

## Failure modes in the output

1. **Missing edges.** Most common. `callees_of(foo)` returns a short, plausible-looking list that happens to be incomplete. There is no warning.
2. **Under-resolved names.** Targets appear as unqualified names or `<builtin>.*` placeholders when pycg couldn't resolve them. Treat these as "call site exists, target unknown."
3. **Hard failures.** On certain decorator stacks, match statements, or walrus patterns, pycg raises and `pycg-mcp build` aborts with a traceback. Not silent — you will see it.

## Guidance for agentic use

- **Good questions for pycg-mcp:**
  - "Who calls this helper I'm about to change?"
  - "What does this function reach, to scope a refactor?"
  - "Which modules import this module?"

- **Bad questions for pycg-mcp:**
  - "Is this function dead code?" (a string-keyed registry may call it)
  - "Is this code path unreachable?" (dynamic dispatch may hit it)
  - "List every place that could invoke this LLM tool." (only the model knows)

- **When the graph looks too sparse:** the code is probably heavy on registries, tool-use, or metaclass magic. Fall back to grep for the function name — a textual search will find string-keyed call sites that static analysis missed.

- **When in doubt:** combine pycg-mcp results with a grep pass before acting on the assumption that something is unused.

## Background

pycg's approach is documented in the original ICSE 2021 paper, *"PyCG: Practical Call Graph Generation in Python."* The limitations above are inherent to static analysis of a dynamically-typed language, not bugs in pycg — no static tool can resolve what only the interpreter knows.
