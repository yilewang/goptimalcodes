#!/bin/bash
makedir=$(ls /home/lmm200007/goptimal/CN/ | cut -f 1 -d '.')

for i in $makedir
do 
    seq -f "python go_sim_hpc.py CN $i %g" 0.01 0.001 0.06 >> jobfile
done

