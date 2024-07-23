# Execution Engine
Generating input trace of LLMServingSim

## `codelet_src`
Module that compiles the model.
`compile.py` uses codelet_src to compile the model and save the model in `compiled_result`.

## `simulate.py`
Where simulation of compiled result is occured. The trace is sotred in `simulator_result`.

**You can add your own simulator here.**

## `full_model.py`
Merges the multiple divided model layers according to the model's structure.

## `swap.py`
Swaps out the attention layer according to the configuration.

## `wrapper`
Where various helper functions are located.
