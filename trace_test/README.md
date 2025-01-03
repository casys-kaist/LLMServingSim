# Testing generate_trace.py

Test your `generate_trace.py` using `test_gen_tace.py`

Modify `main` function of  `test_gen_tace.py` according to your model and hardware.

## Check these informations while testing your function

- Make sure ATTENTION layer is well separated for each request

- Make sure ith layer output and i+1th layer input size are matched

- Make sure ALLREDUCE operation is well placed for synchronization
