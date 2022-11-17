import os, sys
from run import Run

# input files
rwf = "ca2+.rwf"     # name of rwf file
log = "ca2+.log"     # name of log file
module = False         # True if module load g16
nproc = 6          # number of processes

g16root = ""
if not module:
    g16root = "/home/liaocan8/bin/g16/"

calc = Run(rwf, log, module, nproc, g16root)
calc.start()
calc.clean()
