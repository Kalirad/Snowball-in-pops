"""
A individual-based model of sequence divergence on a holey fitness landscape based on RNA
folding to simulate the accumulation of Dobzhansky-Muller incompatibilities
(DMIs).
"""

__author__ = 'Ata Kalirad, Ricardo B. R. Azevedo'

__version__ = '1.0'

import os
import pickle
from copy import *
from itertools import *

import numpy as np
import pandas as pd

# ViennaRNA package python bindings
import RNA as RNA

# global
RNA_nucl = ['A', 'C', 'G', 'U']

# dict storing the secondary structures of RNA sequences during divergence
RNA_folding_dict = {}

# set random number generator seed
np.random.set_state(('MT19937', np.array([
    3691495208, 2881111814, 3977485953,  126022579, 1276930617,
     355103692, 3248493791, 3009808844,  612188080,  248004424,
    1489588601,  173474438, 4039752635, 2508845774, 2622234337,
    2700397831, 1811893199, 2190136060, 2315726008, 1162460778,
    2341168633,  236659960, 3175264097, 3400454537,  427729918,
    4066770621,  567157494, 4014767970, 2930740323,  378300123,
    2705662117, 3891078126, 1960649845, 3044656210,  882045208,
    1570375463, 2086686192,  407452463, 2030931525, 2734889467,
    3712254193, 3949803070,  764947052, 2833180084, 2612938943,
    3513858645, 1012338082, 1723965053,   40253333, 3097240011,
    3472905330,  563287754,  704858225,  610145833, 2824639775,
    3671030693,  225662685, 4093017874,  488496843, 3011853058,
    3141429748, 2892388748, 1752852512, 1097583623, 3335701968,
    2741138771, 2366687650, 2909722827, 3896701472, 2855844360,
      14740992,  126288255,  556395335, 3606698449, 1990092369,
    1892289888, 1025326265, 3335170268, 2955298765, 2086040311,
    2644433388, 1986237624,  831065590, 2567078834, 3535829239,
    1597256603,  781977323, 2945733169, 3479378352, 3652557111,
    1100223342,  235212556, 2599186570,  899620665,  675417868,
    1297279698, 3980368873, 1671894382, 3219957975,  129492647,
     369423255, 1887390651,  536695139, 3467326731,  577893063,
    3628585169, 2772043849,  369219244, 1271097627, 1346409244,
    2331891903,   39930497, 2068899034,  539572370, 4195007861,
    3495378688, 3377756157, 2835342219, 3699793011, 3321615441,
    2211559076, 2398792755, 2796307031,  818646352,  355446500,
    2946711801, 1049957619,  561188288, 2829760282,   55894884,
    1568204679, 1764468784, 1959965565, 4065967902, 3887804509,
    3833073650, 3717783102, 1837449653,  528963116, 4121548680,
    2402147957, 2202929313,  747086954, 3205182257, 1631864764,
     858833100,  148465241,   17458708, 2148761251, 3002919548,
    3773743659, 2611894356, 2275521209, 3027905006, 2234470309,
    2709870512, 1052969526, 3035329785,  110428213, 2893701759,
    2512125031, 3045322315, 2322452091, 3576747394, 2006737455,
     124047895, 3870223050, 3757797920,  698743578,  701653240,
    3561309206,   39541368, 2659965257, 3356207001,  698671102,
    1967130233, 3584965340, 3302789650,  104792115,  989737788,
    1289315250, 2742066874,  943135962, 2610987463, 4156696495,
    1957093316, 1880989243,  211024555, 1594171485, 2646518040,
    1391570537, 2982210346, 3225750783, 1452478140, 1063288625,
    2782363442,  333182057, 2864780704, 3890295634, 1022925971,
     226535384, 2132360150,   74977604, 4208008791, 1697651592,
    4029637378,  397828762, 2954491996, 1120498466, 3197759375,
    2646537589, 2903140119,  580234113, 2324229766, 1485090247,
    3173462698, 1441000100, 3212564317,  598271368, 1052134622,
    2751284206, 4040281713, 2630844601, 1921303308,  861775468,
    3522939180, 2855935558, 3227004083, 4121725263,  805407916,
    1207185676,  785322196, 3104463214, 3070205549, 1984686779,
       5199855, 2585264490, 3703002136, 3352578045,  257641487,
    1613285168, 3845545412, 2884412656, 3795140597, 2864082431,
    1708426814,  661272124, 3359489670, 2989690080, 1120054048,
    3029239860, 2037244341, 3411962036, 3468887812, 1294329307,
    1967939294, 1668712931, 1560596708, 2986374405, 3266952874,
    1758277657, 3876598642, 1149698899, 1548677880, 2464327872,
     466262570, 2573332645, 3577605405, 3511489634, 3001210402,
    4047160993, 1096981688, 1365437714,  967187969, 2651685599,
    4258218418,  618336653, 1813338507, 4161534170, 1206855048,
    3766692676, 1984622584, 1256641952, 2293866774, 2566572107,
    1296931689,  202959755, 3331103372, 3095866549, 1832670718,
    3588629070,  533366259,  301078755, 1299816886, 2612908898,
    1142385071, 4044229138,  392786907, 1473264101,  171872184,
    2873022820, 1878820461,   88690985, 3019565333, 2121461097,
    1522107992, 1733374438, 2311932879,  556408593, 1461835210,
    1423528436,  819211315,  889069790, 3086689727, 1730639543,
    1216615289, 2492159266, 1809961698, 1659780200, 3125102201,
    1711752707, 2723337471, 2521518355, 3884672928, 1313721188,
    1901655237, 3962083231,  757934816, 2196008247, 2111842931,
    2965600004, 1312840433, 3455017541,  545137641, 2279641585,
    2939005091, 1537081838, 2463922331, 1996015762, 1196027276,
     906621855, 1704400250,   76236737,  136244169,  619138087,
      98595120,  719278264, 1334390246, 3171154143, 1280182795,
    2215843496, 2676742417, 2197843524, 1396698993,  609335212,
     723295525, 3605167513, 4155694342, 3017089897, 1955520678,
    4067049686, 3239743094, 1221155545, 4095319239,  425400349,
    1806147353, 3671105575,  627163234, 1861707767,  274296576,
     638507216, 1649469686,  608691281, 4232809768,  611030651,
     853789168, 1733062866,  540453354,   11996619, 2695864391,
    2050310856,  141509199,  252149019, 3547463915,  329855083,
    2856249739, 3735981321, 2875626876, 2379144635,   13062386,
    1562227109, 1191505353, 3203340427, 2778675184, 2770557127,
    3644383877, 1790071106, 2240228460, 1676798968,  863141840,
    1175886689, 1178806726,  358678487, 3328835908, 2633561969,
    4074930335,  772447425, 3430950121, 3352113867,  701629620,
      25420967, 3791888554, 1412926413,  791735289,  161600651,
     506627594, 4220683170,  539553216,  176491711,  870303302,
    2405928427,  673609577,  616683903, 2009922698, 2088461621,
     631204850,  495792565, 1105952597, 1332646700,   23124919,
    2539330986, 1231655942, 1860851071, 3651186456, 2775290123,
    3681258608,  637100105, 4220549846, 3186875083, 3856908269,
    3867761132, 3985657986, 4173577631,  552539584, 2204479092,
    4165177831, 2396591349, 3474222162, 2920321345, 3906718099,
     515536637,  991766590, 2116510279,  482084635, 4005496942,
     374235227, 1711760850, 3750465691,  101652558, 3589303631,
    1360138030, 1382922742,  340163774, 2692240084, 2626346609,
    3041178492, 3616792294,  699158099, 1180482576, 3504356230,
    1897868877,  464615571, 3149754153, 2219112250, 2421357980,
    3182082688, 3145015709, 2579307737, 3490881071, 2970802492,
    3235037551, 1994684505,  355293861, 2682386071, 1408942224,
    3272168205, 3715571520,  476379336, 3644917929,  666542692,
    2680727545,  560661664, 1022989241,  806139402,  495605276,
     462775794, 2795097035, 1348129402, 4137368209, 2768709750,
    2129930451,  422284347, 1297682726, 1252742143, 3031031382,
      75134366, 3411139976, 1654986716,  532012083, 1253013106,
    1814002341,  584805750, 4151151859,  279516416, 2068669679,
    1452548111,  255585988, 2731877417,  805942443, 3209104026,
    1105115396, 1929339947, 3829736722, 2980275336, 2169476831,
     784792828, 3572862771, 1057808935, 1774004947, 3086076921,
     969435958, 4291618669,  892653473, 2713995907, 2137887400,
    2565641007, 1417836736,  415508859, 1624683723,   23763112,
     518111653, 2355447857, 2023934715,  934168085, 2250448450,
     450387908, 1069332538, 4170085337, 2145735300, 2298032455,
    1437026749, 2863147795, 3273446986, 1979692197, 3208629490,
    2080357079,  584771674, 1802076639, 2018580439, 4261231470,
    1708636029, 3602321445,   18885205, 1940272685, 4187271341,
    1647123067, 1450487947, 3463781280, 3759557524,  493883757,
    3901885447, 3190687437,  742916954, 3176758487, 3010187255,
     936898923, 1805555016, 1981187811, 1196213096, 3067885662,
    2550095824, 3396199635, 3614915928, 1977375679, 2173583078,
    2643789240, 2587955166, 2158941995, 2347766906, 1711205114,
      66633020, 3977356823, 1510661526, 3048960083,   51672689,
    3596587592, 4038438382, 4019922490, 2146383929, 1692948176,
    1233895739, 3938222851, 2698966080, 2950467396, 1878048591,
    3547155317, 3627364723,  906814924, 1075129814, 3302437944,
    2756803960, 2719380291, 1774084191, 2789415893, 4095723844,
    1297221824, 1938199324, 4112704123, 1741415251, 1105144176,
    1259977468,  131064353, 4036118418,  311279014], dtype=np.uint32),
    624, 0, 0.0))


class RNASeq(object):
    """RNA sequence and corresponding minimum free energy (MFE) secondary
    structure.

    Attributes
    ----------
    L : int
        Sequence length.
    bp : int
        Number of base pairs.
    mfe : float
        MFE.
    seq : str
        Sequence.
    struct : str
        MFE secondary structure.
    """

    def __init__(self, seq):
        """Initialize RNASeq object from RNA sequence.

        Note
        ----
        Does not check whether sequence contains invalid nucleotides.

        Parameters
        ----------
        seq : str
            Sequence.

        Examples
        --------
        >>> seq = RNASeq('UAAAUCGGCGUUCUGCCACAGCAGCGAA')
        >>> seq.struct
        '........(((((((...))).))))..'
        >>> seq.mfe
        -4.199999809265137
        >>> seq.bp
        7
        """
        self.seq = seq
        self.L = len(seq)
        self.update_struct()
        self.bp = self.struct.count('(')

    def update_struct(self):
        """Calculate MFE and secondary structure of current RNA sequence.
        Check if MFE and secondary structure have already been calculated.

        Note
        ----
        Uses ViennaRNA package.
        """
        if self.seq in RNA_folding_dict:
            self.struct, self.mfe = RNA_folding_dict[self.seq]
        else:
            mfe_struct = RNA.fold(self.seq)
            self.struct, self.mfe = mfe_struct
            RNA_folding_dict[self.seq] = mfe_struct

    @staticmethod
    def random_sequence(L):
        """Generate a random RNA sequence of a certain length.

        Parameters
        ----------
        L : int
            Sequence length.

        Returns
        -------
        RNASeq
            Random sequence.

        Examples
        --------
        >>> L = 100
        >>> unfolded = RNASeq('.' * L)
        >>> seq = RNASeq.random_sequence(L)
        >>> seq.bp == seq.get_bp_distance(unfolded)
        True
        """
        seq_array = np.random.choice(RNA_nucl, size=L)
        return RNASeq(''.join(seq_array))

    @staticmethod
    def convertor(seq, inv=False):
        if inv:
            dic = {0:'A', 1:'U', 2:'C', 3:'G'}
        else:
            dic = {'A':0, 'U':1, 'C': 2, 'G': 3}
        temp = []
        for i in seq:
            temp.append(dic[i])
        if type(temp[0]) == str:
            return ''.join(temp)
        else:
            return temp

    @staticmethod
    def mutate(seq, site, nucl):
        """Mutate RNA sequence at specific site(s) to specific nucleotide(s).

        Note
        ----
        Does not calculate structure of mutated sequence.

        Parameters
        ----------
        seq : str
            RNA sequence.
        site : int or list
            Site(s).
        nucl : str or list
            Nucleotide(s).

        Returns
        -------
        str
            Mutated sequence

        Examples
        --------
        >>> RNASeq.mutate('AAAAAAA', [0, 1, 4], ['C', 'G', 'U'])
        'CGAAUAA'
        """
        if type(site) == list:
            assert len(site) == len(nucl)
            for i, j in zip(site, nucl):
                seq = seq[:i] + j + seq[i + 1:]
            return seq
        else:
            return seq[:site] + nucl + seq[site + 1:]

    @staticmethod
    def mutate_random(seq):
        """Mutate RNA sequence at a single randomly chosen site to a randomly
        chosen nucleotide.

        Note
        ----
        Does not calculate structure of mutant RNA sequence.

        Parameters
        ----------
        seq : str
            Sequence.

        Returns
        -------
        str
            Mutant RNA sequence.
        """
        site = np.random.randint(0, len(seq))
        nucl = [i for i in RNA_nucl if i != seq[site]]
        np.random.shuffle(nucl)
        return RNASeq.mutate(seq, site, nucl[0])

    @staticmethod
    def get_hamdist(seqA, seqB):
        """Calculate Hamming distance between two sequences.

        Parameters
        ----------
        seqA : str
            Sequence.
        seqB : str
            Sequence.

        Returns
        -------
        int
            Hamming distance.

        Examples
        --------
        >>> seqA = 'UAAAUCGGCGUCCUGCCACAGCAGCGAA'
        >>> seqB = 'UAAAUCGGCGUUCUGCCACAGCAGCGAA'
        >>> RNASeq.get_hamdist(seqA, seqB)
        1
        """
        assert type(seqA) == type(seqB)
        diffs = 0
        for ch1, ch2 in zip(seqA, seqB):
            if ch1 != ch2:
                diffs += 1
        return diffs

    def diverge(self, other):
        """Mutate RNASeq while preventing multiple-hits.  Calculate new
        structure.

        Exclude sites that have mutated previously by preventing convergence
        with another sequence.  Throw exception if sequences are already
        completely diverged.

        Parameters
        ----------
        other : RNASeq
            RNA sequence self is diverging from.

        Raises
        ------
        ValueError
            Sequence cannot diverge any more.
        """
        ini_dist = self.get_hamdist(self.seq, other.seq)
        if ini_dist == self.L:
            raise ValueError('The two sequences are already completely diverged.')
        fin_dist = ini_dist
        while fin_dist != ini_dist + 1:
            mut = RNASeq.mutate_random(self.seq)
            fin_dist = self.get_hamdist(mut, other.seq)
        self.seq = mut
        self.update_struct()

    @staticmethod
    def get_diverged_sites(seq1, seq2):
        """Find diverged sites between two sequences.

        Parameters
        ----------
        seq1 : str
            Sequence.
        seq2 : str
            Sequence.

        Returns
        -------
        tuple
            Three elements:
            seq1_nucls: list
                Diverged nucleotides on seq1.
            seq2_nucls: list
                Diverged nucleotides on seq2.
            sites: list
                sites of diverged sites.

        Examples
        --------
        >>> seqA = 'UAAAUCGGCGUCCUGCCACAGCAGCGAA'
        >>> seqB = 'UAAAUCGGCGUUCUGCCACAGCAGCGAA'
        >>> RNASeq.get_diverged_sites(seqA, seqB)
        (['C'], ['U'], [11])
        """
        assert len(seq1) == len(seq2)
        seq1_nucls = []
        seq2_nucls = []
        sites = []
        for i in range(len(seq1)):
            if seq1[i] != seq2[i]:
                seq1_nucls.append(seq1[i])
                seq2_nucls.append(seq2[i])
                sites.append(i)
        return seq1_nucls, seq2_nucls, sites

    @staticmethod
    def introgress(recipient, donor, n_introgr):
        """Construct all possible introgressions of between 1 and 4 diverged
        nucleotides from a donor sequence to a recipient sequence.

        Parameters
        ----------
        recipient : str
            Sequence.
        donor : str
            Sequence.
        n_introgr : int
            Number of nucleotides to introgress.

        Returns
        -------
        list
            Introgressed genotypes.

        Examples
        --------
        >>> RNASeq.introgress('AAAAAAA', 'AAUAUAU', 2)
        ['AAUAUAA', 'AAUAAAU', 'AAAAUAU']
        """
        assert len(recipient) == len(donor)
        assert 1 <= n_introgr <= 4
        indices = []
        temp = RNASeq.get_diverged_sites(recipient, donor)
        if temp != 0:
            indices = temp[2]
        introgress = []
        if n_introgr == 1:
            for i in range(len(recipient)):
                if recipient[i] != donor[i]:
                    introgress.append(recipient[:i] + donor[i] +
                        recipient[i + 1:])
        else:
            indices_introg = list(combinations(indices, n_introgr))
            for i in indices_introg:
                if n_introgr == 2:
                    introgress.append(recipient[:i[0]] + donor[i[0]] +
                        recipient[i[0] + 1:i[1]] + donor[i[1]] +
                        recipient[i[1] + 1:])
                elif n_introgr == 3:
                    introgress.append(recipient[:i[0]] + donor[i[0]] +
                        recipient[i[0] + 1:i[1]] + donor[i[1]] +
                        recipient[i[1] + 1:i[2]] + donor[i[2]] +
                        recipient[i[2] + 1:])
                elif n_introgr == 4:
                    introgress.append(recipient[:i[0]] + donor[i[0]] +
                        recipient[i[0] + 1:i[1]] + donor[i[1]] +
                        recipient[i[1] + 1:i[2]] + donor[i[2]] +
                        recipient[i[2] + 1:i[3]] + donor[i[3]] +
                        recipient[i[3] + 1:])
        return introgress

    @staticmethod
    def recombine(seq1, seq2):
        """Generate all recombinants of two RNA sequences resulting from a
        single crossover event.

        For a sequence of length L, there will be 2 * (L - 1) recombinants.

        Parameters
        ----------
        seq1 : str
            Sequence.
        seq2 : str
            Sequence.

        Returns
        -------
        list
            Recombinants.

        Examples
        --------
        >>> RNASeq.recombine('AAAA', 'UUUU')
        ['AUUU', 'UAAA', 'AAUU', 'UUAA', 'AAAU', 'UUUA']
        """
        assert len(seq1) == len(seq2)
        recs = []
        for i in np.arange(1, len(seq1), 1):
            recs.append(seq1[:i] + seq2[i:])
            recs.append(seq2[:i] + seq1[i:])
        return recs

    @staticmethod
    def get_neighbors(seq):
        """Generate all the mutations needed to specify all single mutation
        neighbors of a sequence.

        Parameters
        ----------
        seq : str
            Sequence.

        Returns
        -------
        seqs : list
            A list of tuples of the form (site, nucleotide).

        Examples
        --------
        >>> RNASeq.get_neighbors('AAA')
        [(0, 'C'), (0, 'G'), (0, 'U'), (1, 'C'), (1, 'G'), (1, 'U'), (2, 'C'), (2, 'G'), (2, 'U')]
        """
        seqs = []
        for i in range(len(seq)):
            for j in RNA_nucl:
                if j != seq[i]:
                    seqs.append((i, j))
        return seqs

    def get_bp_distance(self, other):
        """Calculate base-pair distance between the secondary structures of
        two RNA sequences.

        Note
        ----
        Uses ViennaRNA package.

        Parameters
        ----------
        other : RNASeq

        Returns
        -------
        int
            Base-pair distance.

        Examples
        --------
        >>> seq1 = RNASeq('UAAAUCGGCGUCCUGCCACAGCAGCGAA')
        >>> seq2 = RNASeq('UAAAUCGGCGUUCUGCCACAGCAGCGAA')
        >>> seq1.get_bp_distance(seq2)
        13
        """
        assert type(self) == type(other)
        return RNA.bp_distance(self.struct, other.struct)


class Population(object):
  
    def __init__(self, seq, pop_size, mut_rate=1e-3, rec_rate=0.0, alpha=12):
        """Initialize Population object from reference RNA sequence and alpha.

        The reference sequence and alpha define the fitness landscape.  A
        sequence is viable if its secondary structure (1) has more than alpha
        base pairs and (2) is at most alpha base pairs away from the structure
        of the reference sequence; a sequence is inviable otherwise.

        Set ancestor to reference sequence (no burn-in).

        Parameters
        ----------
        seq : str
            Reference sequence.
        alpha : int, optional
            alpha.
        """
        self.ref_seq = RNASeq(seq)
        self.ancestor = RNASeq(seq)
        self.alpha = alpha
        self.N = pop_size
        self.u = mut_rate
        self.r = rec_rate
        self.burnin()
        self.population = [self.ancestor for i in range(self.N)]
        assert self.ref_seq.bp > self.alpha
        assert type(self.N) == int and self.N > 0
        assert type(self.u) == float and 0 <= self.u <= 1.
        assert type(self.r) == float and 0 <= self.r <= 0.5

    def burnin(self, t=200):
        """Evolve ancestor (initially set to equal the reference sequence) for
        t (viable) substitutions allowing multiple hits.  Update ancestor at
        the end.  Set lineages A and B to ancestor.

        Parameters
        ----------
        t : int
            Length of burn-in period.
        """
        count = 0
        while count < t:
            fix = 0
            while not fix:
                mut = deepcopy(self.ancestor)
                mut = RNASeq(RNASeq.mutate_random(mut.seq))
                fix = self.is_viable(mut)
            self.ancestor = mut
            count += 1

    def get_allele_freqs(self, locus, array= True):
        if array:
            allele_freqs = []
            for i in RNA_nucl:
                p = 0
                for j in self.population:
                    if j.seq[locus] == i:
                        p += 1
                allele_freqs.append(float(p)/float(len(self.population)))
            allele_freqs = np.array(allele_freqs)
        else:
            allele_freqs = {}
            for i in RNA_nucl:
                p = 0
                for j in self.population:
                    if j.seq[locus] == i:
                        p += 1
                allele_freqs[i] = (float(p)/float(len(self.population)))
        return allele_freqs
    
    def get_seq_from_list(self, seqs):
        self.population = []
        for i in (seqs):
            self.population.append(RNASeq(i))

    @property
    def gene_diversity(self):
        L = len(self.ancestor.seq)
        H = np.zeros(L)
        for i in range(L):
            H[i] = 1 - np.power(self.get_allele_freqs(i), 2).sum()
        return H

    @property
    def wt_seq(self):
        return max(self.genotypes_dic, key=self.genotypes_dic.get)

    @property
    def genotype_freqs(self):
        freqs = []
        seqs = [i.seq for i in self.population]
        for i in self.genotypes:
            freqs.append(float(seqs.count(i))/float(len(seqs)))
        return np.array(freqs)

    @property
    def genotypes(self):
        return set([i.seq for i in self.population])

    @property
    def genotypes_dic(self):
        dic = {}
        seqs = [i.seq for i in self.population]
        for i in self.genotypes:
            dic[i] = float(seqs.count(i))/float(len(seqs))
        return dic

    @property
    def pop_bp_distance(self):
        return np.array([RNA.bp_distance(self.ref_seq.struct, i.struct) for i in self.population])

    def is_viable(self, seq):
        """Evaluate whether a sequence is viable

        See Evolve.__init__() for more details.

        Parameters
        ----------
        seq : RNASeq
            Sequence.

        Returns
        -------
        bool
            Viability.
        """
        assert type(self.ref_seq) == type(seq)
        if seq.bp <= self.alpha:
            return False
        else:
            bp = RNASeq.get_bp_distance(self.ref_seq, seq)
            if bp <= self.alpha:
                return 1
            else:
                return 0
    
    def get_pop_hamdist(self, other):
        assert type(other) == Population
        seqsA = []
        seqsB = []
        dist_A = []
        dist_B = []
        dist = []
        for i in self.genotypes_dic:
            if self.genotypes_dic[i] >= self.freq_limit:
                seqsA.append(i)
        if len(seqsA) == 0:
            seqsA.append(max(self.genotypes_dic, key=self.genotypes_dic.get))
        pairs = list(combinations(seqsA, 2))
        for i in pairs:
            dist_A.append(RNASeq.get_hamdist(i[0], i[1]))
        for i in other.genotypes_dic:
            if other.genotypes_dic[i] >= other.freq_limit:
                seqsB.append(i)
        if len(seqsB) == 0:
            seqsB.append(max(other.genotypes_dic, key=other.genotypes_dic.get))
        pairs = list(combinations(seqsB, 2))
        for i in pairs:
            dist_B.append(RNASeq.get_hamdist(i[0], i[1]))
        seqs_comb = seqsA + seqsB
        pairs = list(combinations(seqs_comb, 2))
        for i in pairs:
            if (i[0] in seqsA and i[1] in seqsB) or (i[0] in seqsB and i[1] in seqsA):
                dist.append(RNASeq.get_hamdist(i[0], i[1]))
        if len(dist_A) or len(dist_B):
            mean_dist = np.mean(dist_A + dist_B)
        else:
            mean_dist = 0
        return {'dist_within': mean_dist, 'dist_between': np.mean(dist)}

    @staticmethod
    def get_int_rep(genotypes):
        return np.array([RNASeq.convertor(i.seq) for i in genotypes])

    @staticmethod
    def recombine(N, r, pop1, pop2=0):
        if type(pop1) == type(pop2):
            mat_1 = pop1[np.random.randint(pop1.shape[0], size= pop1.shape[0]/2), :]
            mat_2 = pop2[np.random.randint(pop2.shape[0], size= pop2.shape[0]/2), :]
        else:
            mat_1 = pop1[np.random.randint(pop1.shape[0], size= N/2), :]
            mat_2 = pop1[np.random.randint(pop1.shape[0], size= N/2), :]
        rec_probs = np.random.binomial(1, r, (mat_1.shape[0], mat_1.shape[1] - 1))
        recs = []
        for i,j,k in zip(mat_1, mat_2, rec_probs):
                rec1 = []
                rec2 = []
                rec1.append(i[0])
                rec2.append(j[0])
                site = 1
                for z in k:
                    if z:
                        rec1.append(j[site])
                        rec2.append(i[site])
                    else:
                        rec1.append(i[site])
                        rec2.append(j[site])
                    site += 1
                recs.append(rec1)
                recs.append(rec2)
        return np.array(recs)

    @staticmethod
    def gen_mut_matrix(L, N, u):
        mut_matrix = np.empty((0, L))
        while len(mut_matrix) < N:
            mut = np.random.binomial(1, u, L)
            mut_loci = np.where(mut == 1)
            mut_matrix = np.vstack((mut_matrix, mut))
        return mut_matrix

    @staticmethod
    def mutate(mat, u, n_alleles=4):
        """

        mat : population matrix
        u : mutation rate
        """
        if u > 0:
            mut = np.random.binomial(1, u, size=(mat.shape)) # mutations per site for the entire population
            muts = []
            for i,j in zip(mut, mat):
                temp = []
                for k in range(len(i)):
                    if i[k]:
                        new_allele = np.random.randint(0, n_alleles)
                        while new_allele == j[k]:
                            new_allele = np.random.randint(0, n_alleles)
                        temp.append(new_allele)
                    else:
                        temp.append(j[k])
                muts.append(temp)
            return np.array(muts)
        else:
            return mat

    @staticmethod
    def generate_offspring(genotypes, u, r, N):
        int_rep = Population.get_int_rep(genotypes)
        recombinants = Population.recombine(N, r, int_rep)
        mutants = Population.mutate(recombinants, u)
        return [RNASeq.convertor(i, inv=True) for i in mutants]

    def get_next_generation(self, sexual=True):
        if sexual:
            r = self.r
        else:
            r = 0.0
        offspring = self.generate_offspring(self.population, self.u, r, self.N)
        viable_offspring = [i for i in offspring if self.is_viable(RNASeq(i))]
        self.next_gen_population = [RNASeq(i) for i in viable_offspring]
        self.population = deepcopy(self.next_gen_population)

class TwoPops(object):

    def __init__(self, pop, mig_rate=0):
        self.init_pop = deepcopy(pop)
        self.mig_rate = mig_rate

    def init_history(self):
        """Initialize attributes that will be used to keep a record of the
        evolutionary history during evolution.
        """
        self.divergence = []
        self.lin1 = []
        self.lin2 = []
        self.lin1_bp = []
        self.lin2_bp = []
        self.avg_mfe = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.single = []
        self.double = []
        self.triple = []
        self.viable_sin_pair = []
        self.p1_inv = []
        self.p2_inv = []
        self.p3_inv = []
        self.single_inv = []
        self.double_inv = []
        self.triple_inv = []
        self.viable_sin_pair_inv = []
        self.RI_max = []
        self.RI = []
        self.GST = []
        self.HT = []
        self.HS = []
        self.D = []
        
    def update_history(self):
        """Update evolutionary history.
        """
        self.lin1.append(self.pop1.wt_seq)
        self.lin2.append(self.pop2.wt_seq)
        self.lin1_bp.append(RNASeq(self.pop1.wt_seq).get_bp_distance(self.pop1.ref_seq))
        self.lin2_bp.append(RNASeq(self.pop2.wt_seq).get_bp_distance(self.pop1.ref_seq))
        self.avg_mfe.append(np.mean((RNASeq(self.pop1.wt_seq).mfe, RNASeq(self.pop2.wt_seq).mfe)))
        self.divergence.append(RNASeq.get_hamdist(self.pop1.wt_seq, self.pop2.wt_seq))
        # Introgression 2 -> 1
        temp = self.introgression_assay()
        self.p1.append(temp['p1'])
        self.p2.append(temp['p2'])
        self.p3.append(temp['p3'])
        self.single.append(temp['single'])
        self.double.append(temp['double'])
        self.triple.append(temp['triple'])
        self.viable_sin_pair.append(temp['viable_sin_pair'])
        # Introgression 1 -> 2
        temp = self.introgression_assay(dir=2)
        self.p1_inv.append(temp['p1'])
        self.p2_inv.append(temp['p2'])
        self.p3_inv.append(temp['p3'])
        self.single_inv.append(temp['single'])
        self.double_inv.append(temp['double'])
        self.triple_inv.append(temp['triple'])
        self.viable_sin_pair_inv.append(temp['viable_sin_pair'])
        # RI
        temp = self.get_RI()
        self.RI_max.append(temp['RI_max'])
        self.RI.append(temp['RI'])
        # population genetic measures
        temp = self.get_genetic_variation()
        self.GST.append(temp['GST'])
        self.HT.append(temp['HT'])
        self.HS.append(temp['HS'])
        self.D.append(temp['D'])

    @property
    def stats(self):
        """Generate a dictionary of evolutionary history attributes.

        Used by save_stats().

        Returns
        -------
        dict
            Evolutionary history.
        """
        stats = {}
        stats['ancestor'] = self.pop1.ancestor.seq
        stats['ref_seq'] = self.pop1.ref_seq.seq
        stats['divergence'] = self.divergence
        stats['seqs_lin1'] = self.lin1
        stats['seqs_lin2'] = self.lin2
        stats['lin1_bp'] = self.lin1_bp
        stats['lin2_bp'] = self.lin2_bp
        stats['mfe_avg'] = self.avg_mfe
        stats['p1'] = self.p1
        stats['p2'] = self.p2
        stats['p3'] = self.p3
        stats['single'] = self.single
        stats['double'] = self.double
        stats['triple'] = self.triple
        stats['viable_sin_pair'] = self.viable_sin_pair
        stats['p1_inv'] = self.p1_inv
        stats['p2_inv'] = self.p2_inv
        stats['p3_inv'] = self.p3_inv
        stats['single_inv'] = self.single_inv
        stats['double_inv'] = self.double_inv
        stats['triple_inv'] = self.triple_inv
        stats['viable_sin_pair_inv'] = self.viable_sin_pair_inv
        stats['RI_max'] = self.RI_max
        stats['RI'] = self.RI
        stats['GST'] = self.GST
        stats['HT'] = self.HT
        stats['HS'] = self.HS
        stats['D'] = self.D
        return stats

    def save_stats(self, directory, file_ID):
        """Save evolutionary history in pickle format.

        Parameters
        ----------
        directory : str
            Directory.
        file_ID : int
            A number added to the file name to avoid overwriting.

        Returns
        -------
        name : pickle
            Pickled stat dictionary.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        file = open(directory + "/stats_" + str(file_ID), 'w')
        pickle.dump(self.stats, file)
        file.close()

    def migrate(self):
        pop1 = [i.seq for i in self.pop1.population]
        pop2 = [i.seq for i in self.pop2.population]
        migrants = np.random.poisson(self.mig_rate)
        if len(pop1) <= len(pop2):
            indices = [np.random.randint(0, len(pop1)) for i in range(migrants)]
            while len(set(indices)) < migrants:
                indices = [np.random.randint(0, len(pop1)) for i in range(migrants)]
        else:
            indices = [np.random.randint(0, len(pop2)) for i in range(migrants)]
            while len(set(indices)) < migrants:
                indices = [np.random.randint(0, len(pop2)) for i in range(migrants)]
        new_pop1 = []
        new_pop2 = []
        for i in range(len(pop1)):
            if i in indices:
                new_pop1.append(pop2[i])
            else:
                new_pop1.append(pop1[i])
        for i in range(len(pop2)):
            if i in indices:
                new_pop2.append(pop1[i])
            else:
                new_pop2.append(pop2[i])
        self.pop1.get_seq_from_list(new_pop1)
        self.pop2.get_seq_from_list(new_pop2)

    def get_inviable_introgressions(self, recipient, donor, n_introgr):
        """Find inviable introgressions from donor sequence to recipient
        sequence.

        Parameters
        ----------
        recipient : str
            Sequence.
        donor : str
            Sequence.
        n_introgr : int
            Number of nucleotides to introgress.

        Returns
        -------
        dict
            Inviable introgressions; keys are tuples containing site numbers.
        """
        introgressions = RNASeq.introgress(recipient, donor, n_introgr)
        inviable = {}
        for i in introgressions:
            seq = RNASeq(i)
            if not self.pop1.is_viable(seq):
                sites = RNASeq.get_diverged_sites(recipient, i)[2]
                key = tuple(sites)
                inviable[key] = ([donor[site] for site in sites], self.pop1.ref_seq.get_bp_distance(seq))
        return inviable, len(introgressions)

    def introgression_assay(self, dir=1):
        """Find single, double, and triple introgressions in which result in an inviable genotype

        Parameters
        ----------
        dir : int, optional
            Direction of introgression. '1' tests 2 -> 1 DMIs whereas '2' tests 1 -> 2 DMIs.

        Returns
        -------
        dict
            A dictionary of single, double, and triple introgressions in one direction which result in inviable genotypes.

        Raises
        ------
        ValueError
            dir can only be '1' or '2'.
        """
        if dir == 1:
            seqA = self.pop1.wt_seq
            seqB = self.pop2.wt_seq
        elif dir == 2:
            seqA = self.pop2.wt_seq
            seqB = self.pop1.wt_seq
        else:
            raise ValueError('Must be either 1 or 2.')
        sin, n_sin = self.get_inviable_introgressions(seqA, seqB, 1)
        dou, n_dou = self.get_inviable_introgressions(seqA, seqB, 2)
        tri, n_tri = self.get_inviable_introgressions(seqA, seqB, 3)
        # remove inviable double introgressions that are caused by inviable
        # Single introgressions
        sin_pairs = list(combinations(sin, 2))
        sin_pairs_seqs = [RNASeq.mutate(seqA, [i[0][0], i[1][0]], [sin[i[0]][0][0], sin[i[1]][0][0]]) for i in sin_pairs]
        sin_pairs_seqs_w = [self.pop1.is_viable(RNASeq(i)) for i in sin_pairs_seqs]
        # single introgressions
        trimmed_dou = {}
        for i in dou:
            found = False
            for j in sin:
                if j[0] in i:
                    found = True
                    break
            if not found:
                trimmed_dou[i] = dou[i]
        trimmed_dou
        # remove inviable triple introgressions that are caused by inviable
        # single or double introgressions
        trimmed_tri = {}
        for i in tri:
            found = False
            #print 'tri in sin'
            for j in sin:
                #print j, i
                if j[0] in i:
                    found = True
                    break
            if not found:
                for j in trimmed_dou:
                    if sum([k in j for k in i]) == 2:
                        found = True
                        break
            if not found:
                trimmed_tri[i] = tri[i]

        if n_sin > 0:
            p1 = len(sin) / float(n_sin)
        else:
            p1 = 0
        if n_dou > 0:
            p2 = len(dou) / float(n_dou)
        else:
            p2 = 0
        if n_tri > 0:
            p3 = len(tri) / float(n_tri)
        else:
            p3 = 0
        return {
            'single': sin,
            'double': trimmed_dou,
            'triple': trimmed_tri,
            'p1': p1,
            'p2': p2,
            'p3': p3,
            'viable_sin_pair': np.sum(sin_pairs_seqs_w),
        }

    def rescue_inviable_introgressions(self, dir, putative_21_DMIs, putative_12_DMIs):
        """Find double introgressions from two putative DMIs in lineages 1 and 2 which result in a viable genotype

        Parameters
        ----------
        dir : int
            Direction of introgression. '1' tests 2 -> 1 DMIs whereas '2' tests 1 -> 2 DMIs.
        putative_21_DMIs : dict
            Putative DMIs found through introgression from lineage 2 to 1.
        putative_12_DMIs : dict
            Putative DMIs found through introgression from lineage 1 to 2.

        Returns
        -------
        dict
            A dictionary of putative DMIs that could be rescued via a second introgression.

        Raises
        ------
        ValueError
            dir can only be '1' or '2'.
        """
        if dir == 1:
            seqA = self.pop1.wt_seq
            seqB = self.pop2.wt_seq
            putative_21 = putative_21_DMIs
            putative_12 = putative_12_DMIs
        elif dir == 2:
            seqA = self.pop2.wt_seq
            seqB = self.pop1.wt_seq
            putative_21 = putative_12_DMIs
            putative_12 = putative_21_DMIs
        else:
            raise ValueError('Must be either 1 or 2.')
        rescue = {}
        for i in putative_21:
            site_i = i[0]
            nucl_i = putative_21[i][0][0]
            seq_i = RNASeq.mutate(seqA, site_i, nucl_i)
            fit_i = self.pop1.is_viable(RNASeq(seq_i))
            for j in putative_12:
                site_j = j[0]
                nucl_j = seqB[site_j]
                seq_ij = RNASeq.mutate(seq_i, site_j, nucl_j)
                fit_ij = self.pop1.is_viable(RNASeq(seq_ij))
                if fit_ij:
                    rescue[tuple(np.sort([site_i, site_j]))] = [(site_i, site_j), (nucl_i, nucl_j)]
        return rescue

    def get_inviable_neighbors(self, seq):
        """Get inviable sequences that are a single mutation away from a
        sequence.

        Parameters
        ----------
        seq : str
            Sequence

        Returns
        -------
        int
            Number of inviable neighbors.
        """
        nei = RNASeq.get_neighbors(seq)
        inviable = []
        for site, nucl in nei:
            mut = RNASeq.mutate(seq, site, nucl)
            if not self.pop1.is_viable(RNASeq(mut)):
                inviable.append((site, nucl))
        return inviable

    def get_genetic_variation(self):
        """
        Measure genetic variation in population.

        Output:
        HS: mean gene diversity within demes
        HT: total gene diversity (pooling all demes)
        GST: Nei's GST
        D: Jost's D
        """
        H_within = []
        H_within.append(self.pop1.gene_diversity.mean())
        H_within.append(self.pop2.gene_diversity.mean())
        pooled = [i.seq for i in self.pop1.population] + [i.seq for i in self.pop2.population]
        pooled_pop = Population(self.pop1.ancestor.seq, len(pooled))
        pooled_pop.get_seq_from_list(pooled)
        HS = np.mean(H_within)
        HT = pooled_pop.gene_diversity.mean()
        if HT == 0:
            GST = 0.
        else:
            GST = 1. - (HS / HT)
        D = (HT - HS) / (1 - HS) * 2
        return {
            'GST': GST, 
            'HT': HT, 
            'HS': HS, 
            'D':D
            }

    def get_RI(self):
        recs = RNASeq.recombine(self.pop1.wt_seq, self.pop2.wt_seq)
        recs_w = [self.pop1.is_viable(RNASeq(i)) for i in recs]
        RI_max = 1. - np.sum(recs_w)/float(len(recs_w))
        WH = 0
        WS = 0
        RI_r = 0
        if self.pop1.r:
            int_rep_1 = Population.get_int_rep(self.pop1.population)
            int_rep_2 = Population.get_int_rep(self.pop2.population)
            recs = Population.recombine(self.pop1.N, self.pop1.r, int_rep_1, int_rep_2)
            recs = [inv_convertor(i) for i in recs]
            recs_w = [self.pop1.is_viable(RNASeq(i)) for i in recs]
            WS = np.sum(recs_w)/float(len(recs_w))
            pop1_recs = Population.recombine(self.pop1.N, self.pop1.r, int_rep_1)
            pop1_recs = [inv_convertor(i) for i in pop1_recs]
            pop1_recs_w = [self.pop1.is_viable(RNASeq(i)) for i in pop1_recs]
            WS_1 = np.sum(pop1_recs_w)/float(len(pop1_recs_w))
            pop2_recs = Population.recombine(self.pop2.N, self.pop2.r, int_rep_2)
            pop2_recs = [inv_convertor(i) for i in pop2_recs]
            pop2_recs_w = [self.pop2.is_viable(RNASeq(i)) for i in pop2_recs]
            WS_2 = np.sum(pop2_recs_w)/float(len(pop2_recs_w))
            WS = np.mean([WS_1, WS_2])
            RI_r = 1. - WH/WS
        return {
            'WH' : WH, 
            'WS' : WS, 
            'RI_max': RI_max, 
            'RI': RI_r
            }

    def evolve(self, gen, step=500, verbose=False):
        self.init_history()
        self.t = 0
        self.pop1 = deepcopy(self.init_pop)
        self.pop2 = deepcopy(self.init_pop)
        self.update_history()
        for i in np.arange(1, gen + 1, 1):
            self.t += 1
            self.pop1.get_next_generation()
            self.pop2.get_next_generation()
            if self.mig_rate:
                self.migrate()
            if not i%step:
                self.update_history()
                if verbose:
                    print i,

if __name__ == "__main__":
    import doctest
    doctest.testmod()
