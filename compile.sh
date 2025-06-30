#!/bin/bash

module purge
module load intel fftw

make clean MACHINE=nova

./configure --enable-mpi --enable-shearing-box --enable-fargo --with-particles=feedback --with-gas=hydro --with-eos=isothermal --with-problem=par_strat3d --with-order=3p --enable-fft
#--with-particle-self-gravity=fft_disk

make all MACHINE=nova
