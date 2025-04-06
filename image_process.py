#!/bin/python

from multiprocessing import Pool
import timeit
import sys
import src
import functools

if __name__ == '__main__':
    core_count = 1
    if len(sys.argv) > 1:
        core_count = int(sys.argv[1])
    if len(sys.argv) < 2:
        raise Exception("Too few arguments given")
    print(f'{core_count} was given for arguments {sys.argv[2:]}')
    
    #parse options
    #0: image_process.py
    #1: number of cores
    #2: function name
    #3: Optional -args
    
    fn = sys.argv[2]
    if not hasattr(src, fn):
        raise Exception("\"" + fn +"\" is not a valid function")
    fn = getattr(src, fn)    

    #Split opts from entries
    args = sys.argv[3:]
    opts = []
    for arg in args:
        if arg[0] == '-':
            opts.append(arg)
    args = [ _ for _ in args if _ not in opts ]
    
    #Create options
    plot_args = {}
    if '-centered' in opts:
        plot_args.update({'opt':True})

    #Run
    print(f"Starting with {core_count} counts")
    start = timeit.default_timer()
    
    plotter = functools.partial(fn, **plot_args)
    with Pool(core_count) as p:
        pi_list = (p.map(plotter, args))

    stop = timeit.default_timer()

    print('Time: ', stop - start)
