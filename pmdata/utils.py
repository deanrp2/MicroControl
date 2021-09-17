import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def mult_sym(df):
    """
    Add more samples according to the symmetries in the problem.
    Will end up with 4 times number of samples
    """
    ht = pd.DataFrame(np.zeros_like(df), columns = df.columns, index = [a + "_h" for a in df.index])
    vt = pd.DataFrame(np.zeros_like(df), columns = df.columns, index = [a + "_v" for a in df.index])
    rt = pd.DataFrame(np.zeros_like(df), columns = df.columns, index = [a + "_r" for a in df.index])

    #swap drum positions
    hkey = ["theta" + str(i) for i in np.array([3, 2, 1, 0, 7, 6, 5, 4], dtype = int) + 1]
    vkey = ["theta" + str(i) for i in np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype = int) + 1]
    rkey = ["theta" + str(i) for i in np.array([4, 5, 6, 7, 0, 1, 2, 3], dtype = int) + 1]

    ht.loc[:, hkey] = df.loc[:, ["theta" + str(i+1) for i in range(8)]].values
    vt.loc[:, vkey] = df.loc[:, ["theta" + str(i+1) for i in range(8)]].values
    rt.loc[:, rkey] = df.loc[:, ["theta" + str(i+1) for i in range(8)]].values

    #adjust angles
    ht.loc[:, hkey] = (3*np.pi - ht.loc[:, hkey]) % (2 * np.pi)
    vt.loc[:, vkey] = (2*np.pi - vt.loc[:, hkey]) % (2 * np.pi)
    rt.loc[:, rkey] = (np.pi + rt.loc[:, hkey]) % (2 * np.pi)

    #fill criticalities and runtimes
    ht.iloc[:, :4] = df.iloc[:, :4].values
    vt.iloc[:, :4] = df.iloc[:, :4].values
    rt.iloc[:, :4] = df.iloc[:, :4].values

    #fill quadrant tallies
    hkey = [2, 1, 4, 3]
    vkey = [4, 3, 2, 1]
    rkey = [3, 4, 1, 2]

    for typ in ["flux", "fiss", "fissE"]: #iterate through 8 tally types
        for unc in ["Q", "_runcertQ"]:
            ht.loc[:, [typ + unc + str(i) for i in hkey]] = \
                    df.loc[:, [typ + unc + str(i) for i in range(1, 5)]].values
            vt.loc[:, [typ + unc + str(i) for i in vkey]] = \
                    df.loc[:, [typ + unc + str(i) for i in range(1, 5)]].values
            rt.loc[:, [typ + unc + str(i) for i in rkey]] = \
                    df.loc[:, [typ + unc + str(i) for i in range(1, 5)]].values

    #combine dfs
    combined = pd.concat([df, ht, vt, rt]).sort_index()

    return combined

def g2l(thetas):
    """
    convert from the global coordinate system to the local one described
    in  MicroControl/README.md
    """
    loc_offsets = np.array([3.6820187359906447, 4.067668586955522,
                           2.2155167202240653-np.pi, 2.6011665711889425-np.pi,
                           0.5404260824008517, 0.9260759333657285,
                           5.3571093738138575-np.pi, 5.742759224778734-np.pi])
    thkeys = ["theta%i"%(i+1) for i in range(8)]
    thetas.loc[:, thkeys] = \
            (thetas.loc[:, thkeys] \
            - loc_offsets + 2 * np.pi) #apply corect 0 point
    thetas.loc[:, ["theta"+str(i) for i in [3,4,5,6]]] *= -1 #reverse ncessary angles
    thetas.loc[:, thkeys] = \
            thetas.loc[:, thkeys] % (2*np.pi) #scale all to [0, 2*np.pi]
    for col in thkeys:
        thetas.loc[:, col].loc[thetas.loc[:, col] > np.pi] -= 2*np.pi #bring to [-pi, pi]
    return thetas

def qpower_preprocess(df, sym = True, uncert = False):
    """
    Process csv files into cleaner quadrant power dataframes for easier ML training.
    sym is whether to use symmetry to mutliple samples
    uncert is whether to also pull the uncertainties
    """
    if sym: #apply sample multiplicity if necessary
        df = mult_sym(df)

    #fiter out uneeded columns
    thetacols = ["theta" + str(i) for i in range(1, 9)]
    fluxcols = ["fluxQ" + str(i) for i in range(1, 5)]
    if uncert:
        uncertcols = ["flux_runcertQ" + str(i) for i in range(1, 5)]
        df = df[thetacols + fluxcols + uncertcols]
    else:
        df = df[thetacols + fluxcols]

    #scale flux tallies to sum to 1
    df[fluxcols] = df[fluxcols].div(df[fluxcols].sum(1), axis = "rows")

    #bring coordinates into local system
    df = g2l(df)

    return df

def careful_split(df, tstfrac = False, tstnum = False):
    """
    Returns a train and test split which does not allow samples coming from the same calculation
    to exist across the train/test line. In other words, samples generated from symmetry must stay
    with the sample they were made from.

    df: must be dataframe with the indices given from the mult_sym function. make sure to sort_index
    tstfrac: generate the split based on an approximate fraction to include in the TESTING set
    tstnum: generate the split based on an approximate total number to include in the TESTING set. The argument here
        will be approximately the total number of rows in the testing dataframe that is returned
    """
    tot_samples = df.shape[0]

    #determine size of final testing set divided by 4
    if tstnum:
        tstsize = int(tstnum//4)
    elif tstfrac:
        tstsize = int((tot_samples*tstfrac)//4)
    else:
        raise Exception("Please specify tstfrac or tstsize")

    #get sample names associate with original calcs
    samplenums = np.unique([a[:12] for a in df.index])

    #pick a set from the original sample names
    tidxr = list(np.random.choice(samplenums, tstsize, replace = False))

    #expand set to include multiplied sample names as well
    tidxf = sorted(tidxr + [a + "_h" for a in tidxr] + [a + "_r" for a in tidxr] + [a + "_v" for a in tidxr])

    #index original dataframe to get the sets
    train = df.drop(tidxf)
    test = df.loc[tidxf]

    return train, test

