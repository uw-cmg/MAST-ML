#!/usr/bin/env python
############
# wt% to at% converter
# TTM 2017-01-23 created to explore discrepancy between CD 2017-01 and 
#                CD_LWR_clean08.csv at% values
############
import sys
import pymatgen as pmg
    
inputtypelist = ["atomic","weight"]

def convert(compdict, inputtype="", verbose=2):
    print("Output:")
    outdict=dict()
    denom = 0.0
    if inputtype == "atomic":
        for symbol in compdict.keys():
            denom = denom + (compdict[symbol]["percent"] * compdict[symbol]["atomic_mass"])
        for symbol in compdict.keys():
            numerator = compdict[symbol]["percent"] * compdict[symbol]["atomic_mass"] 
            new_perc = numerator / denom * 100.0
            outdict[symbol]=dict()
            outdict[symbol]["perc_out"]=new_perc
            print("%s: %3.5f weight percent" % (symbol, new_perc))
    elif inputtype == "weight":
        for symbol in compdict.keys():
            denom = denom + (compdict[symbol]["percent"] / compdict[symbol]["atomic_mass"])
        for symbol in compdict.keys():
            numerator = compdict[symbol]["percent"] / compdict[symbol]["atomic_mass"] 
            new_perc = numerator / denom * 100.0
            outdict[symbol]=dict()
            outdict[symbol]["perc_out"]=new_perc
            print("%s: %3.5f atomic percent" % (symbol, new_perc))
    else:
        print("Input type %s not yet supported." % inputtype)
        return None
    return outdict

def remaining_percent(compdict):
    """Adds up percentages and calculates remaining percent.
        Args:
            compdict <dict>: Dictionary by symbol of fractions
    """
    perc_tot=0.0
    for symbol in compdict.keys():
        perc_tot = perc_tot + compdict[symbol]["percent"]
    remainder = 100.0 - perc_tot
    return remainder

def parse_composition(compraw="", fillwith="Fe", verbose=2):
    """
        Args:
            compraw <str>: String of atomic symbols and compositions,
                            separated by spaces and commas, e.g. 
                            Cu 0.23, Ni 0.25
            fillwith <str>: Atomic symbol of element to fill in composition
                            if percents do not add up to 100.
                            Default Fe
            verbose <int>: 0 - silent
                            1 - verbose
                            2 - debug
    """
    complist = compraw.strip().split(",")
    if len(complist) == 0:
        print("No composition given.")
        return None
    compdict=dict()
    for cidx in range(0, len(complist)):
        csplit = complist[cidx].strip().split()
        symbol = csplit[0].strip()
        element = pmg.core.periodic_table.Element(symbol)
        atomic_mass = element.atomic_mass
        percent = float(csplit[1].strip())
        compdict[symbol]=dict()
        compdict[symbol]['element'] = element
        compdict[symbol]['atomic_mass'] = atomic_mass #in amu
        compdict[symbol]['percent'] = percent
    remainder = remaining_percent(compdict)
    if remainder > 0.000000:
        if verbose > 0 :
            print("Filling in remaining %3.3f percent with %s." % (remainder, fillwith))
        element = pmg.core.periodic_table.Element(fillwith)
        atomic_mass = element.atomic_mass
        compdict[fillwith]=dict()
        compdict[fillwith]['element'] = element
        compdict[fillwith]['atomic_mass'] = atomic_mass #in amu
        compdict[fillwith]['percent'] = remainder
    if verbose > 1:
        for symbol in compdict.keys():
            print("%s, %3.3f amu, %3.3f input percent" % (symbol, compdict[symbol]["atomic_mass"],compdict[symbol]["percent"]))
    return compdict

def main(compraw="", inputtype="", verbose=2):
    """
        Args:
            compraw <str>: String of atomic symbols and compositions,
                            separated by commas, e.g. Cu, 0.23, Ni, 0.25
            inputtype <str>: Percentage type of input
    """
    if not (inputtype in inputtypelist):
        print("%s is not an option in %s" % (inputtype, inputtypelist))
        return None
    compdict = parse_composition(compraw)
    if verbose > 1:
        print("Input:")
        for symbol in compdict.keys():
            print("%s: %3.5f %s percent" % (symbol, compdict[symbol]["percent"], inputtype))
        print("")
    outdict = convert(compdict, inputtype)
    if outdict == None:
        return
    if verbose > 1:
        tot_new_perc=0.0
        for symbol in outdict.keys():
            tot_new_perc = tot_new_perc + outdict[symbol]["perc_out"]
        print("Total: %3.3f" % tot_new_perc)
    return

if __name__ == "__main__":
    #python percent_converter.py "Ni 80, Cr 20" weight
    main(sys.argv[1], sys.argv[2])
