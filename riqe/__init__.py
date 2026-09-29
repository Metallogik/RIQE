"""RIQE, Radiology Image Quality Evaluator: a NIQE-style no-reference quality
model for CT.

Note on parallelism. The scripts of this project use one process per slice.
If every process lets numpy/OpenBLAS open its own threads, a 32-core machine
reaches a load of 90 and the work slows down instead of speeding up
(measured). The scripts therefore set the environment to one thread per
process **before** importing numpy.
"""
