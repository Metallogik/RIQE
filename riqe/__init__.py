"""RIQE - valutatore di qualita' no-reference per immagini TC.

Nota sul parallelismo.  Gli script di questo progetto usano un processo per
slice.  Se ogni processo lascia che numpy/OpenBLAS apra i propri thread, su
una macchina a 32 core si arriva a un load di 90 e il lavoro rallenta invece
di accelerare: misurato.  Gli script impostano quindi le variabili
d'ambiente a un thread per processo **prima** di importare numpy.
"""
