#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inneholder nyttige funksjoner og klasser som ikke passer inn i de andre modulene.
"""

import time
import locale

class Stoppeklokke:
    """
    Kontekstsetter til å måle tidsbruk med.
    
    Kilde:https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python?answertab=votes#tab-top

    Bruk
    ```
    >>> with Stoppeklokke("Sumutregning"):
    ...    print(sum(range(100_000)))  # doctest: +ELLIPSIS
    4999950000
    Sumutregning: ... s
    
    ```
        
    """
    def __init__(self, navn=''):
        """
        Opprett stoppeklokke med navn
        
        :param str navn: Navn til stoppeklokka. 
        Brukes for å skrive ut resultatet. Standard: ''
        """
        self.navn = navn

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        tid = time.time() - self.tstart
        print(f"{self.navn}: {locale.str(tid)} s")


if __name__ == '__main__':
    import doctest
    doctest.testmod()