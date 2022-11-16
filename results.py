import numpy as np
import utilities as util

class Results:
    def __init__(self, load, core):
        self.Load = load
        self.Core = core

    def print_results(self, ifOscStr=True):
        t_start = util.start_time("print_results")
        with open("soc_energy.txt", "w") as f:
            f.write("Shown below is the excitation energy of the SOC states expressed as a linear combination of "
                    "non-SOC states.\n\n")

            for state in range(2*self.Load.NStates+1):
                f.write("SOC State: " + str(state) + "    Excitation Energy: " \
                        + '%.4f' % round(self.Core.ExEnergy_soc[state]*util.Hartree2eV, 4))

                if ifOscStr and state > 0:
                    f.write("    Osc Str: " + '%.4f' % round(self.Core.osc_str_soc[state], 4))

                f.write("\n")

                for c in range(2*self.Load.NStates+1):
                    re_contrib = np.real(self.Core.states_soc[c, state])
                    im_contrib = np.imag(self.Core.states_soc[c, state])
                    contrib = re_contrib**2 + im_contrib**2

                    if contrib > 1e-6:
                        f.write("State: " + str(c) + " (Gaussian Index: " + str(self.Load.old_index[c]) + ")      " \
                                + '%.4f' % round(contrib*100, 4) + "%\n")

                f.write("\n")

        util.end_time("print_results", t_start)