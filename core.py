import pandas as pd
import sys
import numpy as np
import utilities as util
import multiprocessing as mp
from itertools import permutations

class Core:
    def __init__(self, load, nproc):
        self.Load = load
        self.hx = None
        self.hy = None
        self.hz = None
        self.Hso = np.zeros((2*self.Load.NStates+1, 2*self.Load.NStates+1), dtype="complex128")
        self.ExEnergy_soc = None
        self.states_soc = None
        self.osc_str_soc = None
        self.nproc = nproc


    def compound_index(self, i, a):
        return self.Load.NVirt*i+a-self.Load.NOcc


    def boettger(self):
        t_start = util.start_time("boettger_factor")
        Vx = self.Load.Vx
        Vy = self.Load.Vy
        Vz = self.Load.Vz
        Q = lambda l: l * (l + 1) * (2 * l + 1) / 3

        for mu in range(1, Vx.shape[0]+1, 1):
            l_mu = self.Load.AO[mu][1]
            center_mu = self.Load.AO[mu][0]
            atom_mu = self.Load.atoms[center_mu]

            for nu in range(1, Vx.shape[1]+1, 1):
                l_nu = self.Load.AO[nu][1]
                center_nu = self.Load.AO[nu][0]
                atom_nu = self.Load.atoms[center_nu]

                self.Load.Vx[mu-1, nu-1] = Vx[mu-1, nu-1] * (1 - np.sqrt(Q(l_mu)*Q(l_nu)/(atom_mu*atom_nu)))
                self.Load.Vy[mu-1, nu-1] = Vy[mu-1, nu-1] * (1 - np.sqrt(Q(l_mu) * Q(l_nu) / (atom_mu * atom_nu)))
                self.Load.Vz[mu-1, nu-1] = Vz[mu-1, nu-1] * (1 - np.sqrt(Q(l_mu) * Q(l_nu) / (atom_mu * atom_nu)))

        util.end_time("boettger_factor", t_start)


    def Hso_eq(self, bk):
        B = bk[0]
        spinB = self.Load.spin[B]
        oldB = self.Load.old_index[B]
        msB = self.Load.Ms[B]

        K = bk[1]
        spinK = self.Load.spin[K]
        oldK = self.Load.old_index[K]
        msK = self.Load.Ms[K]

        t_start = util.start_time(f"Hso equation B = {B}, K = {K}")

        sqrt2 = np.sqrt(2)
        final_answer = None

        #  Excited State Coupling
        if B != 0 and K != 0:
            normB = np.linalg.norm(self.Load.XPY[:, oldB - 1])
            normK = np.linalg.norm(self.Load.XPY[:, oldK - 1])
            if normB * normK < 1E-07:
                sys.exit("Vector for excited state" + str(oldB) + " or " + str(oldK) + " has 0 norm.")

            #  Excited Singlet and Triplet Coupling
            #  only <S|Hso|T> cases are treated explicitly
            if spinB < spinK:

                #  Case where Ms = +/- 1
                if abs(msK) == 1:

                    termji = 0.0 + 0.0j
                    for i in range(self.Load.NOcc):
                        for j in range(self.Load.NOcc):
                            for a in range(self.Load.NOcc, self.Load.NBsUse):
                                termji += self.Load.XPY[self.compound_index(i, a), oldB - 1] \
                                          * self.Load.XPY[self.compound_index(j, a), oldK - 1] \
                                          * (self.hx[j, i] + 1j * self.hy[j, i] * msK)

                    termab = 0.0 + 0.0j
                    for a in range(self.Load.NOcc, self.Load.NBsUse):
                        for b in range(self.Load.NOcc, self.Load.NBsUse):
                            for i in range(self.Load.NOcc):
                                termab += self.Load.XPY[self.compound_index(i, a), oldB - 1] \
                                          * self.Load.XPY[self.compound_index(i, b), oldK - 1] \
                                          * (self.hx[a, b] + 1j * self.hy[a, b] * msK)

                    term = (termji - termab) / (2 * sqrt2 * normB * normK * msK)
                    final_answer = term

                # Case where Ms = 0
                elif msK == 0:

                    termji = 0.0 + 0.0j
                    for i in range(self.Load.NOcc):
                        for j in range(self.Load.NOcc):
                            for a in range(self.Load.NOcc, self.Load.NBsUse):
                                termji += self.Load.XPY[self.compound_index(i, a), oldB - 1] \
                                          * self.Load.XPY[self.compound_index(j, a), oldK - 1] \
                                          * self.hz[j, i]

                    termab = 0.0 + 0.0j
                    for a in range(self.Load.NOcc, self.Load.NBsUse):
                        for b in range(self.Load.NOcc, self.Load.NBsUse):
                            for i in range(self.Load.NOcc):
                                termab += self.Load.XPY[self.compound_index(i, a), oldB - 1] \
                                          * self.Load.XPY[self.compound_index(i, b), oldK - 1] \
                                          * self.hz[a, b]

                    term = (termab - termji) / (2 * normB * normK)
                    final_answer = term

            #  Same Spin Coupling
            elif spinB == spinK:

                #  Singlet-Singlet Coupling is Zero
                if spinB == 0:
                    final_answer = 0.0 + 0.0j

                #  Triplet-Triplet Coupling
                elif spinB == 1:

                    #  Coupling between Ms = 0 and Ms = 0, along with Ms = +/- 1 and Ms = -/+ 1
                    if (msB + msK) == 0:
                        final_answer = 0.0 + 0.0j

                    #  Coupling between Ms = +/- 1 and Ms = +/- 1
                    elif abs(msK + msB) == 2:

                        termji = 0.0 + 0.0j
                        for i in range(self.Load.NOcc):
                            for j in range(self.Load.NOcc):
                                for a in range(self.Load.NOcc, self.Load.NBsUse):
                                    termji += self.Load.XPY[self.compound_index(i, a), oldB - 1] \
                                              * self.Load.XPY[self.compound_index(j, a), oldK - 1] \
                                              * self.hz[j, i]

                        termab = 0.0 + 0.0j
                        for a in range(self.Load.NOcc, self.Load.NBsUse):
                            for b in range(self.Load.NOcc, self.Load.NBsUse):
                                for i in range(self.Load.NOcc):
                                    termab += self.Load.XPY[self.compound_index(i, a), oldB - 1] \
                                              * self.Load.XPY[self.compound_index(i, b), oldK - 1] \
                                              * self.hz[a, b]

                        term = (termji + termab) / (2 * normB * normK * msB)
                        final_answer = term

                    #  Coupling between Ms = 0 and Ms = +/- 1
                    elif msB == 0:
                        if abs(msK) == 1:

                            termji = 0.0 + 0.0j
                            for i in range(self.Load.NOcc):
                                for j in range(self.Load.NOcc):
                                    for a in range(self.Load.NOcc, self.Load.NBsUse):
                                        termji += self.Load.XPY[self.compound_index(i, a), oldB - 1] \
                                                  * self.Load.XPY[self.compound_index(j, a), oldK - 1] \
                                                  * (self.hx[j, i] + 1j * self.hy[j, i] * msK)

                            termab = 0.0 + 0.0j
                            for a in range(self.Load.NOcc, self.Load.NBsUse):
                                for b in range(self.Load.NOcc, self.Load.NBsUse):
                                    for i in range(self.Load.NOcc):
                                        termab += self.Load.XPY[self.compound_index(i, a), oldB - 1] \
                                                  * self.Load.XPY[self.compound_index(i, b), oldK - 1] \
                                                  * (self.hx[a, b] + 1j * self.hy[a, b] * msK)

                            term = (termji + termab) / (2 * sqrt2 * normK * normB)
                            final_answer = term

        #  Ground-Excited State Coupling
        #  Only treating <GS|Hso|EX> terms explicitly
        elif B == 0:

            #  Ground State Singlet - Excited State Singlet Coupling
            if spinK == 0:
                final_answer = 0.0 + 0.0j

            #  Ground State Singlet - Triplet Coupling
            elif spinK == 1:

                normK = np.linalg.norm(self.Load.XPY[:, oldK - 1])
                if normK < 1E-07:
                    sys.exit("Vector for excited state " + str(oldK) + " has 0 norm.")

                #  For Ms = 0 Triplet
                if msK == 0:

                    termia = 0.0 + 0.0j
                    for i in range(self.Load.NOcc):
                        for a in range(self.Load.NOcc, self.Load.NBsUse):
                            termia += self.Load.XPY[self.compound_index(i, a), oldK - 1] \
                                      * self.hz[i, a]

                    term = termia / (sqrt2 * normK)
                    final_answer = term

                # For Ms = +/- 1
                elif abs(msK) == 1:

                    termia = 0.0 + 0.0j
                    for i in range(self.Load.NOcc):
                        for a in range(self.Load.NOcc, self.Load.NBsUse):
                            termia += self.Load.XPY[self.compound_index(i, a), oldK - 1] \
                                      * (self.hx[i, a] + self.hy[i, a] * 1j * msK)

                    term = -termia / (2 * normK * msK)
                    final_answer = term

        util.end_time(f"Hso equation B = {B}, K = {K}", t_start)
        return bk, final_answer


    def build_Hso(self):
        t_start = util.start_time("build_Hso")
        self.hx = np.matmul(self.Load.MO.T, np.matmul(self.Load.Vx, self.Load.MO))
        self.hy = np.matmul(self.Load.MO.T, np.matmul(self.Load.Vy, self.Load.MO))
        self.hz = np.matmul(self.Load.MO.T, np.matmul(self.Load.Vz, self.Load.MO))

        bkpairs = list(permutations(range(2*self.Load.NStates+1), 2))
        bklist = [bk for bk in bkpairs\
                  if ((bk[0] != 0 and bk[1] != 0) \
                      and ((self.Load.spin[bk[0]] <= self.Load.spin[bk[1]]) and self.Load.spin[bk[1]] == 1)) \
                  or (bk[0] == 0 and self.Load.spin[bk[1]] == 1)]

        with mp.Pool(processes=self.nproc) as pool:
            p = pool.map_async(self.Hso_eq, bklist)
            for result in p.get():
                if result[1] is not None:
                    self.Hso[result[0][0], result[0][1]] = result[1]
                    self.Hso[result[0][1], result[0][0]] = np.conj(result[1])
                else:
                    pass

        ReHso = self.Hso.real + np.diag(self.Load.ExEnergy)
        ImHso = self.Hso.imag

        ReHso[abs(ReHso) < 1e-7] = 0
        ImHso[abs(ImHso) < 1e-7] = 0

        self.Hso = ReHso + ImHso*1j

        pd.DataFrame(self.Hso.round(5)).to_csv("Hso.csv")
        util.end_time("build_Hso", t_start)


    def calc_soc(self):
        t_start = util.start_time("calc_soc")
        self.ExEnergy_soc, self.states_soc = np.linalg.eigh(self.Hso)
        util.end_time("calc_soc", t_start)


    def calc_osc_str(self):
        t_start = util.start_time("build_Hso")
        dip_soc = np.matmul(self.Load.dip.T, self.states_soc)
        osc_str_soc = [0]
        for state in range(1, 2*self.Load.NStates+1):
            f = np.real((2.0/3.0)*self.ExEnergy_soc[state]*np.dot(dip_soc[:, state].conjugate(), dip_soc[:, state]))
            osc_str_soc.append(f)
        self.osc_str_soc = osc_str_soc
        util.end_time("build_Hso", t_start)



