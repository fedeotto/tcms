# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 10:49:04 2022

@author: fedeotto
"""
import collections
import re
import smact
from smact import Element, Species, screening
from smact.oxidation_states import Oxidation_state_probability_finder
from smact.screening import pauling_test
import itertools
import numpy as np

all_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
               'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
               'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
               'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
               'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
               'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
               'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
               'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
               'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
               'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
               'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
               'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
        
class CompositionError(Exception):
    """Exception class for composition errors"""
    pass


def get_sym_dict(f, factor):
    sym_dict = collections.defaultdict(float)
    for m in re.finditer(r"([A-Z][a-z]*)\s*([-*\.\d]*)", f):
        el = m.group(1)
        amt = 1
        if m.group(2).strip() != "":
            amt = float(m.group(2))
        sym_dict[el] += amt * factor
        f = f.replace(m.group(), "", 1)
    if f.strip():
        raise CompositionError(f'{f} is an invalid formula!')
    return sym_dict

def parse_formula(formula):
    '''
    Parameters
    ----------
        formula: str
            A string formula, e.g. Fe2O3, Li3Fe2(PO4)3.
    Return
    ----------
        sym_dict: dict
            A dictionary recording the composition of that formula.
    Notes
    ----------
        In the case of Metallofullerene formula (e.g. Y3N@C80),
        the @ mark will be dropped and passed to parser.
    '''
    # for Metallofullerene like "Y3N@C80"
    formula = formula.replace('@', '')
    formula = formula.replace('[', '(')
    formula = formula.replace(']', ')')
    m = re.search(r"\(([^\(\)]+)\)\s*([\.\d]*)", formula)
    if m:
        factor = 1
        if m.group(2) != "":
            factor = float(m.group(2))
        unit_sym_dict = get_sym_dict(m.group(1), factor)
        expanded_sym = "".join(["{}{}".format(el, amt)
                                for el, amt in unit_sym_dict.items()])
        expanded_formula = formula.replace(m.group(), expanded_sym).replace('"', '')
        return parse_formula(expanded_formula)
    sym_dict = get_sym_dict(formula, 1)
    return sym_dict


def _fractional_composition(formula):
    elmap = parse_formula(formula)
    elamt = {}
    natoms = 0
    for k, v in elmap.items():
        if abs(v) >= 1e-6:
            elamt[k] = v
            natoms += abs(v)
    comp_frac = {key: elamt[key] / natoms for key in elamt}
    return comp_frac


def _fractional_composition_L(formula):
    comp_frac = _fractional_composition(formula)
    atoms = list(comp_frac.keys())
    counts = list(comp_frac.values())
    return atoms, counts


def _element_composition(formula):
    elmap = parse_formula(formula)
    elamt = {}
    natoms = 0
    for k, v in elmap.items():
        if abs(v) >= 1e-6:
            elamt[k] = v
            natoms += abs(v)
    return elamt

def _element_composition_L(formula):
    comp_frac = _element_composition(formula)
    atoms = list(comp_frac.keys())
    counts = list(comp_frac.values())
    return atoms, counts

def extract_oxidation_state(element, included_species):
    pattern = re.compile(f'{element}(-?\\d*)')
    oxidation_states = []

    for elem in included_species:
        match = pattern.search(elem)
        if match:
            oxidation_state = int(match.group(1)) if match.group(1) else 1
            oxidation_states.append(oxidation_state)

    return oxidation_states

def check_formula(formula: str     = 'Na0.5Cl0.5',
                  charge_threshold = 1e-3,
                  pauling_test     = True,
                  verbose          = True):
    
    '''Checks formula based on oxidation states and pauling test.'''
    
    common_ox_states = {
        #first column
        'H': 1, 'Li': [1], 'Na': [1], 'K' : [1], 'Rb': [1], 'Cs': [1], 'Fr': [1],

        #second column
        'Be': [2], 'Mg': [2], 'Ca': [2], 'Sr': [2], 'Ba': [2], 'Ra': [2],

        #common cations
        'Al': [3], 'Ga': [3], 'In': [3], 'Tl': [1,3], 'B': [3],
        'Ti': [4], 'Zr': [4], 'Hf': [4], 'V' : [4,5], 'Nb': [4,5], 'Ta': [4,5],
        'Si': [-4,4], 'Ge': [4],

        #common anions
        'O' : [-2], 'F' : [-1], 'Cl': [-1], 'Br': [-1], 'I' : [-1],
    }

    #Pauling test: more electronegative elements have lower/negative oxidation states
    elems, ratios = _fractional_composition_L(formula)
    ratios        = np.array(ratios, dtype=np.float32)
    space         = smact.element_dictionary(elems)
    
    smact_elems   = [e[1] for e in space.items()]


    electronegs = [e.pauling_eneg for e in smact_elems]

    for elem in smact_elems:
        if elem.symbol in common_ox_states:
            elem.oxidation_states = common_ox_states[elem.symbol]
    
    #  get all tuples of oxidations states for each element.
    ox_combos     = [e.oxidation_states for e in smact_elems]
    charge_bal    = False
    electroneg_OK = False
                
    for ox_states in itertools.product(*ox_combos):
        charge_bal = np.abs(np.dot(ox_states, ratios)) < charge_threshold
        if charge_bal:
            if pauling_test:
                electroneg_OK = smact.screening.pauling_test(ox_states, electronegs)
                break
        else:
            continue

    if not (charge_bal and electroneg_OK):
        ox_states = 'N/A'
    
    return ox_states, charge_bal, electroneg_OK

#OLD CHECK
def compute_prob(formula: str = 'InO3'):
    '''
    Nonsense detector using oxidation states model 
    proposed in the paper https://pubs.rsc.org/en/content/articlelanding/2018/fd/c8fd00032h

    '''
    #Define ox finder
    try:
        ox_prob_finder   = Oxidation_state_probability_finder()
        included_species = ox_prob_finder.get_included_species()

        elem_dict = _element_composition(formula)
        elems     = list(elem_dict.keys())
        stoichs   = list(elem_dict.values())

        ox_states_dict = dict.fromkeys(elems)
        for el in elems:
            ox_states = extract_oxidation_state(el, included_species)
            ox_states_dict[el] = ox_states
        
        ox_combs     = itertools.product(*ox_states_dict.values()) 
        results      = []

        for ox in ox_combs:
            specie = [Species(e, oxidation=o) for e,o in zip(elems,ox)]    
            prob   = ox_prob_finder.compound_probability(specie)
            results.append((elems, ox, prob))
        
        most_likely    = max(results, key=lambda x: x[2])
        most_likely_ox = most_likely[1]
        is_stable      = not bool(np.dot(np.array(most_likely_ox), np.array(stoichs)))
                #ox_states,     comp_prob       if its stable
        return (most_likely[1], most_likely[2], is_stable)
    except Exception as e:
        print(e)
        return ('N/A', 'N/A', 'N/A')

if __name__ == '__main__':
    result = check_formula('BaTiO3')
    
    #CaNbO3
    #Ti2O3
    #VO2

