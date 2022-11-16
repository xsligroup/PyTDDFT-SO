import os, sys, random, math
import utilities as util
import numpy as np
import pandas as pd
import time

class Load:
    def __init__(self, rwf, log, ifModule, NBasis, NBsUse, NStates, NOcc, NVirt, NTT, g16root=None):
        self.rwf = rwf
        self.log = log
        self.ifModule = ifModule
        self.g16root = g16root
        self.NBasis = NBasis
        self.NBsUse = NBsUse
        self.NStates = NStates
        self.NOcc= NOcc
        self.NVirt = NVirt
        self.NTT = NTT
        self.NTTB = int(self.NBsUse * (self.NBsUse + 1) / 2)
        self.NOAVA = int(self.NOcc * self.NVirt)
        self.NAtoms = int(list(filter(None, os.popen("grep 'NAtoms' " + self.log).read().split(" ")))[1])
        self.filno = {"MOcoef": '524R', "SOints": "617R", "XPY": "635R"}
        self.Vx = None
        self.Vy = None
        self.Vz = None
        self.MO = None
        self.XPY = None
        self.XMY = None
        self.X = None
        self.Y = None
        self.osc_str = None
        self.spin = None
        self.Ms = None
        self.ExEnergy = None
        self.old_index = None
        self.atoms = None
        self.AO = None
        self.dip = None


    def parse_rwf(self):
        t_start = util.start_time("parse_rwf")
        if not os.path.exists(self.rwf):
            sys.exit('rwf file does not exist!')

        if not self.ifModule:
            for keys in self.filno.keys():
                os.system(self.g16root + '/rwfdump ' + self.rwf + ' ' + keys + '.tmp ' + self.filno[keys])
        else:
            for keys in self.filno.keys():
                os.system('rwfdump ' + self.rwf + ' ' + keys + '.tmp ' + self.filno[keys])

        length_XPY = util.rwf_length("XPY.tmp", 12)
        subfile_XMY = int(2*self.NOcc*self.NVirt*length_XPY/(4*self.NOcc*self.NVirt + 1))

        data_MO = util.import_data("MOcoef.tmp")
        data_SOints = util.import_data("SOints.tmp")
        data_XPY = util.import_data("XPY.tmp", 12)[:2*self.NOAVA*self.NStates]
        data_XMY = util.import_data("XPY.tmp", 12+subfile_XMY)[:2*self.NOAVA*self.NStates]

        self.MO = np.reshape(data_MO, (self.NBsUse, self.NBasis)).T  # Gaussian stores matrices column major

        self.Vx = util.square(data_SOints[0:self.NTT], True, True) * util.k
        self.Vy = util.square(data_SOints[self.NTT:2*self.NTT], True, True) * util.k
        self.Vz = util.square(data_SOints[2*self.NTT:3*self.NTT], True, True) * util.k

        # Formatting Excitation Amplitude Data
        aux = np.reshape(data_XPY, (2*self.NStates, self.NOAVA))
        self.XPY = np.array([aux[i] for i in range(0, 2*self.NStates, 2)], dtype="float64").T

        aux = np.reshape(data_XMY, (2*self.NStates, self.NOAVA))
        self.XMY = np.array([aux[i] for i in range(0, 2*self.NStates, 2)], dtype="float64").T

        self.X = 0.5 * (self.XPY + self.XMY)
        self.Y = 0.5 * (self.XPY - self.XMY)

        util.end_time("parse_rwf", t_start)


    def parse_log(self):
        t_start = util.start_time("parse_log")
        if not os.path.exists(self.log):
            sys.exit('log file does not exist!')

        # Obtaining excited state information
        os.system("grep 'S\*\*' " + self.log + " >> log.tmp")
        osc_str = []
        old_spin = [0]
        spin = []
        ExEnergy = []
        old_index = []
        Ms = []
        with open("log.tmp", "r") as logtmp:
            lines = logtmp.readlines()
            for l in range(len(lines)):
                l2 = list(filter(None, lines[l].split(" ")))

                spin_str = l2[3].split("-")[0]
                if spin_str == "Triplet":
                    a = 0
                    old_spin.append(1)
                    while a < 3:
                        spin.append(1)
                        old_index.append(l+1)
                        ExEnergy.append(float(l2[4])*util.eV2Hartree)
                        osc_str.append(float(l2[8].split("=")[1]))
                        Ms.append(a-1)
                        a += 1

                elif spin_str == "Singlet":
                    spin.append(0)
                    old_spin.append(0)
                    old_index.append(l+1)
                    ExEnergy.append(float(l2[4]) * util.eV2Hartree)
                    osc_str.append(float(l2[8].split("=")[1]))
                    Ms.append(0)

                elif spin_str == "1.000":
                    sys.exit("Excited State " + str(l+1) + "has undetermined spin.\n Use chemical intuition to manually "
                                                        "assign spin states by editting the log file.")

        self.osc_str = np.array([0.0] + osc_str, dtype="float64")
        self.spin = np.array([0] + spin, dtype=int)
        self.Ms = np.array([0] + Ms, dtype="int")
        self.ExEnergy = np.array([0.0] + ExEnergy, dtype="float64")
        self.old_index = np.array([0] + old_index, dtype="int")

        # Obtaining atomic center, AO, and transition electric dipole moments info
        find_atoms = os.popen("grep 'Input orientation' " + self.log).read()
        find_AO = os.popen("grep 'Molecular Orbital Coefficients:' " + self.log).read()
        find_dip = os.popen("grep 'Ground to excited state transition electric dipole moments (Au)' " + self.log).read()
        atoms = {}
        AO = {}
        dip = np.zeros((2*self.NStates+1, 3))

        with open(self.log, "r") as log:
            lines = log.readlines()

            # Identifying atomic centers
            start_atoms = lines.index(find_atoms) + 5
            for l in lines[start_atoms:start_atoms + self.NAtoms]:
                aux = list(filter(None, l.split(" ")))
                center = int(aux[0])
                atomic_num = int(aux[1])
                atoms[center] = atomic_num

            # Identifying AO center and orbital angular momentum
            start_AO = lines.index(find_AO) + 4
            for l in lines[start_AO:start_AO + self.NBasis]:
                l2 = list(filter(None, l.split(" ")))
                letters = set(util.OAM.keys())

                if len(l2) == 9:
                    current_center = int(l2[1])
                    ang_mom = util.OAM["".join(set(l2[3]).intersection(letters))]
                    AO[int(l2[0])] = (current_center, ang_mom)

                elif len(l2) == 7:
                    ang_mom = util.OAM["".join(set(l2[1]).intersection(letters))]
                    AO[int(l2[0])] = (current_center, ang_mom)

                elif len(l2) == 8:
                    ang_mom = util.OAM["".join(set(l2[1]).intersection(letters))]
                    AO[int(l2[0])] = (current_center, ang_mom)

                else:
                    sys.exit("Trouble parsing AO data:\n " + l)

            start_dip = lines.index(find_dip) + 2
            tracker = 1
            for l in lines[start_dip: start_dip + self.NStates]:
                l2 = list(filter(None, l.split(" ")))
                state = int(l2[0])
                spin = old_spin[state]

                if spin == 0:
                    dip[tracker, 0] = l2[1]
                    dip[tracker, 1] = l2[2]
                    dip[tracker, 2] = l2[3]
                    tracker += 1
                elif spin == 1:
                    dip[tracker, 0] = l2[1]
                    dip[tracker, 1] = l2[2]
                    dip[tracker, 2] = l2[3]
                    dip[tracker + 1, 0] = l2[1]
                    dip[tracker + 1, 1] = l2[2]
                    dip[tracker + 1, 2] = l2[3]
                    dip[tracker + 2, 0] = l2[1]
                    dip[tracker + 2, 1] = l2[2]
                    dip[tracker + 2, 2] = l2[3]
                    tracker += 3

        self.atoms = atoms
        self.AO = AO
        self.dip = dip

        os.system("rm log.tmp")
        util.end_time("parse_log", t_start)


    def create_index_guide(self):
        t_start = util.start_time("create_index_guide")
        new_index = list(range(2 * self.NStates+1))
        pre_df = {"Gaussian Index": self.old_index, "RTDDFT-SO Index":new_index,
                  "Excitation Energy": self.ExEnergy*util.Hartree2eV, "Osc Str": self.osc_str, "Spin": self.spin, "Ms": self.Ms}
#        print("old_index: ", len(self.old_index))
#        print("new_index: ", len(new_index))
#        print("Ex Energy: ", len(self.ExEnergy))
#        print("osc str: ", len(self.osc_str))
#        print("spin: ", len(self.spin))
        df = pd.DataFrame(pre_df)
        with open("indexing_guide.txt", "w") as txt:
            txt.write("This is to orient you to the new indexing presented by RTDDFT-SO.\n")
            txt.write("The states given by this code does not follow the previous excited state numbering presented "
                      "in the Gaussian log file.\n\n")
            txt.write(str(df))
        util.end_time("create_index_guide", t_start)


    def spot_check(self):
        """
        Randomly selects 2 MOs and excited states.
        Checks if the MOs are orthonormal.
        Checks in <X-Y|X+Y> normalization is met.
        Checks in excited states are orthogonal.
        :return: None
        """

        rand_MO1 = random.randint(0, self.NBsUse-1)
        rand_MO2 = random.randint(0, self.NBsUse-1)
        while rand_MO2 == rand_MO1:
            rand_MO2 = random.randint(0, self.NBsUse-1)

        rand_EX1 = random.randint(0, self.NStates-1)
        rand_EX2 = random.randint(0, self.NStates-1)
        while rand_EX1 == rand_EX2:
            rand_EX2 = random.randint(0, self.NStates-1)

        MO1_MO1 = np.linalg.norm(self.MO[:, rand_MO1])
        MO2_MO2 = np.linalg.norm(self.MO[:, rand_MO2])
        MO1_MO2 = np.dot(self.MO[:, rand_MO1], self.MO[:, rand_MO2])

        EX1_EX1 = np.dot(self.XMY[:, rand_EX1], self.XPY[:, rand_EX1])
        EX2_EX2 = np.dot(self.XMY[:, rand_EX2], self.XPY[:, rand_EX2])

        EX1_EX2_X = np.dot(self.X[:, rand_EX1], self.X[:, rand_EX2])
        EX1_EX2_Y = np.dot(self.Y[:, rand_EX1], self.Y[:, rand_EX2])

        print("-----------------------------------------")
        print("Sanity check on imported data from G16.\n ")

        if math.isclose(MO1_MO1, 1, rel_tol=1e-1):
            print("MO " + str(rand_MO1+1) + " is normalized.\n")
        else:
            print("MO " + str(rand_MO1+1) + " is NOT normalized.")
            print("Norm of MO " + str(rand_MO1+1) + ": " + str(MO1_MO1) + "\n")

        if math.isclose(MO2_MO2, 1, rel_tol=1e-1):
            print("MO " + str(rand_MO2+1) + " is normalized.\n")
        else:
            print("MO " + str(rand_MO2+1) + " is NOT normalized.")
            print("Norm of MO " + str(rand_MO2+1) + ": " + str(MO2_MO2) + "\n")

        if math.isclose(MO1_MO2, 0.0, abs_tol=1e-3):
            print("MO " + str(rand_MO1+1) + " and " + str(rand_MO2+1) + " are orthogonal.\n")
        else:
            print("MO " + str(rand_MO1+1) + " and " + str(rand_MO2+1) + " are not orthogonal.")
            print("Product of MO " + str(rand_MO1+1) + " and MO " + str(rand_MO2+1) + ": " + str(MO1_MO2) + "\n")

        if math.isclose(EX1_EX1, 0.5, rel_tol=1e-2):
            print("Excited State " + str(rand_EX1+1) + " meets the <X-Y|X+Y> = 0.5 normalization condition.\n")
        else:
            print("Excited State " + str(rand_EX1+1) + " does NOT meet the <X-Y|X+Y> = 0.5 normalization condition.")
            print("Excited State " + str(rand_EX1+1) + " <X-Y|X+Y> = " + str(EX1_EX1) + "\n")

        if math.isclose(EX2_EX2, 0.5, rel_tol=1e-2):
            print("Excited State " + str(rand_EX2+1) + " meets the <X-Y|X+Y> = 0.5 normalization condition.\n")
        else:
            print("Excited State " + str(rand_EX2+1) + " does NOT meet the <X-Y|X+Y> = 0.5 normalization condition.")
            print("Excited State " + str(rand_EX2+1) + " <X-Y|X+Y> = " + str(EX2_EX2) + "\n")

        if math.isclose(EX1_EX2_X - EX1_EX2_Y, 0.0, abs_tol=1e-3):
            print("MO " + str(rand_EX1+1) + " and " + str(rand_EX2+1) + " are orthogonal.\n")
        else:
            print("MO " + str(rand_EX1+1) + " and " + str(rand_EX2+1) + " are not orthogonal.")
            print("Product of MO " + str(rand_EX1) + " and MO " + str(rand_EX2) + ": " + str(EX1_EX2_X - EX1_EX2_Y) + "\n")



















