import os
from read_g16 import Load
from core import Core
from results import Results
import time
from datetime import datetime
import concurrent.futures as cf

class Run:
    def __init__(self, rwf, log, ifModule, nproc, g16root=None):
        self.rwf = rwf
        self.log = log
        self.ifModule = ifModule
        self.g16root = g16root
        self.NBasis = int(list(filter(None, os.popen("grep 'NBF=' " + self.log).read().split(" ")))[1])
        self.NBsUse = int(list(filter(None, os.popen("grep 'NBFU' " + self.log).read().split(" ")))[1])
        self.NStates = int(list(filter(None, os.popen("grep 'Excited State' " + self.log).read().split()))[-8][:-1])
        self.NOcc = int(list(filter(None, os.popen("grep 'alpha electrons' " + self.log).read().split()))[0])
        self.NVirt = int(self.NBsUse - self.NOcc)
        self.NTT =  int(self.NBasis * (self.NBasis + 1) / 2)
        self.nproc = nproc


    def check_memory(self):
        SOints_mem = (self.NBsUse ** 2) * 24 + self.NTT * 24
        MOcoef_mem = self.NBsUse * self.NBasis * 8
        Omega_mem = self.NStates * 16
        Hso_mem = ((self.NStates * 2 + 1) ** 2) * 16
        f_mem = self.NStates * 16
        EVec_mem = ((self.NStates * 2 + 1) ** 2) * 16
        total_mem = (SOints_mem + MOcoef_mem + Omega_mem + Hso_mem + f_mem + EVec_mem) * 1.0E-09
        """
        go = input("The calculation requires at least " + str(total_mem) + " GB of memory. Do you want to continue? Yes or No: ")
        while go not in ['Yes', 'yes', 'no', 'No']:
            go = input(
                "The calculation requires at least " + str(total_mem) + " of memory. Do you want to continue? Yes or No: ")
        if go == 'no' or go == 'No':
            sys.exit("Job too big for ya big boy?")
        elif go == 'yes' or go == 'Yes':
            pass
        else:
            sys.exit('Must be yes or no to continue.')
        """

    def start(self):
        diary = open("diary.txt", "w")
        diary.write(f"R-TDDFTSO started on {datetime.now()}\n\n")
        self.start_time = time.perf_counter()
        diary.close()

        #self.check_memory()
        data = Load(self.rwf, self.log, self.ifModule, self.NBasis, self.NBsUse, self.NStates,
                    self.NOcc, self.NVirt, self.NTT, self.g16root)

        with cf.ThreadPoolExecutor() as executor:
            f1 = executor.submit(data.parse_rwf)
            f2 = executor.submit(data.parse_log)

        data.create_index_guide()

        # data.spot_check()

        calculate = Core(data, self.nproc)
        calculate.boettger()
        calculate.build_Hso()
        calculate.calc_soc()
        calculate.calc_osc_str()

        results = Results(data, calculate)
        results.print_results(ifOscStr=True)

        diary = open("diary.txt", "a")
        diary.write(f"R-TDDFTSO ended on {datetime.now()}\n")
        elapsed_time = time.perf_counter() - self.start_time
        diary.write(f"R-TDDFTSO calculation took {elapsed_time}\n")
        diary.close()


    def clean(self):
        os.system("rm *.tmp")
