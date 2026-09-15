"""LLMServingSim simulator internals.

Each module here owns one piece of the per-iteration loop driven by
``serving.__main__``. Imports inside this subpackage use relative form
(``from .X import ...``); external callers use ``from serving.core.X import ...``.
"""
