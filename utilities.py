import itertools
import numpy as np
from datetime import datetime
import time

eV2Hartree = 0.0367493 # from Google
Hartree2eV = 27.2114 # from Google

k = 0.0072973525**2 / 2


def rwf_length(file, trim=0):
    """
    Finds the length of rwf file with offset included.
    :param file: Name of text file where the rwfdump data is stored
    :param trim: offset on reading the data
    :return: int: length of file
    """
    linenum = 0
    array = []
    with open(file, "r") as tmp:
        lines = tmp.readlines()
        for l in lines:
            if l.startswith(" Dump of file"):
                break
            else:
                linenum += 1

        length = int(list(filter(None, lines[linenum].split(" ")))[-5]) - trim
        return length


def import_data(file, trim=0):
    """
    Find the data stored in a text file obtained after rwfdump.
    The data is imported as a 1D array.
    :param file: Name of text file where the rwfdump data is stored
    :param trim: int: offset on reading the data
    :return: 1D Numpy array of data
    """
    linenum = 0
    array = []
    with open(file, "r") as tmp:
        lines = tmp.readlines()
        for l in lines:
            if l.startswith(" Dump of file"):
                break
            else:
                linenum += 1

        aux = []
        for l in lines[linenum+1:]:
            newl = l.replace("\n", " ")
            newl2 = newl.replace("D", "E")
            aux.append(list(filter(None, newl2.split(" "))))
        array_str = list(itertools.chain.from_iterable(aux))
        array = [float(x) for x in array_str[trim:]]
    return np.array(array)


def square(LTM, ifHermitian=True, makeComplex=False):
    """
    Turns row major lower triangle matrix represented in a 1D array to square matrix
    :param LTM: Lower triangular matrix stored in a row major 1D Numpy array
    :param ifHermitian: True if the full matrix is Hermitian, False if anti-Hermitian
    :param makeComplex: True if data is stored as real but is truly complex
    :return: Numpy array of the full square matrix
    """
    dim = int((np.sqrt(8*LTM.shape[0]+1)-1)/2)
    aux = []
    prev = 0
    for i in range(dim):
        aux.append(list(LTM[prev:prev+i+1]))
        prev += i+1

    for row in aux:
        while len(row) < dim:
            row.append("0")

    M = np.array(aux,dtype="float64")

    if makeComplex:
        M = M.astype("complex128") * 1j
        if ifHermitian:
            M = M + M.T.conjugate() - np.diag(np.diag(M))
        elif not ifHermitian:
            M = M + M.T - np.diag(np.diag(M))
        M = -M

    elif not makeComplex:
        M = M + M.T - np.diag(np.diag(M))

    return M


def list_breaker(l, n):
    """
    Partitions a list into a list of lists with size < n
    :param l: list
    :param n: max elements per sub-list
    :return: partitioned list
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def start_time(f_name):
    """
    Writes the start time in diary.txt and returns a timestamp of when this was called.
    :param f_name: The process this is measuring
    :return:
    """
    diary = open("diary.txt", "a")
    abs_time = datetime.now()
    diary.write(f"{f_name} Start Time: {abs_time}\n\n")
    t_start = time.perf_counter()
    diary.close()
    return t_start


def end_time(f_name, t_start):
    """
    Writes the amount of time that has passed since t_start
    :param f_name:
    :param t_start:
    :return:
    """
    diary = open("diary.txt", "a")
    t_end = time.perf_counter()
    t_elapsed = t_end - t_start
    diary.write(f"Time elapsed for {f_name}: {t_elapsed}\n\n")
    diary.close()


OAM = {
    "S": 0,
    "P": 1,
    "D": 2,
    "F": 3,
    "G": 4,
    "H": 5,
    "I": 6
}

ptable = {
1:"H",
2:"He",
3:"Li",
4:"Be",
5:"B",
6:"C",
7:"N",
8:"O",
9:"F",
10:"Ne",
11:"Na",
12:"Mg",
13:"Al",
14:"Si",
15:"P",
16:"S",
17:"Cl",
18:"Ar",
19:"K",
20:"Ca",
21:"Sc",
22:"Ti",
23:"V",
24:"Cr",
25:"Mn",
26:"Fe",
27:"Co",
28:"Ni",
29:"Cu",
30:"Zn",
31:"Ga",
32:"Ge",
33:"As",
34:"Se",
35:"Br",
36:"Kr",
37:"Rb",
38:"Sr",
39:"Y",
40:"Zr",
41:"Nb",
42:"Mo",
43:"Tc",
44:"Ru",
45:"Rh",
46:"Pd",
47:"Ag",
48:"Cd",
49:"In",
50:"Sn",
51:"Sb",
52:"Te",
53:"I",
54:"Xe",
55:"Cs",
56:"Ba",
57:"La",
58:"Ce",
59:"Pr",
60:"Nd",
61:"Pm",
62:"Sm",
63:"Eu",
64:"Gd",
65:"Tb",
66:"Dy",
67:"Ho",
68:"Er",
69:"Tm",
70:"Yb",
71:"Lu",
72:"Hf",
73:"Ta",
74:"W",
75:"Re",
76:"Os",
77:"Ir",
78:"Pt",
79:"Au",
80:"Hg",
81:"Tl",
82:"Pb",
83:"Bi",
84:"Po",
85:"At",
86:"Rn",
87:"Fr",
88:"Ra",
89:"Ac",
90:"Th",
91:"Pa",
92:"U",
93:"Np",
94:"Pu",
95:"Am",
96:"Cm",
97:"Bk",
98:"Cf",
99:"Es",
100:"Fm",
101:"Md",
102:"No",
103:"Lr",
104:"Rf",
105:"Db",
106:"Sg",
107:"Bh",
108:"Hs",
109:"Mt",
110:"Ds",
111:"Rg",
112:"Cn",
113:"Nh",
114:"Fl",
115:"Mc",
116:"Lv",
117:"Ts",
118:"Og",
}
