import numpy as np
from path import Pathlib

def mult_sym(df):
    """
    Add more samples according to the symmetries in the problem.
    Will end up with 4 times number of samples
    """
    ht = pd.DataFrame(np.zeros_like(df), columns = df.columns, index = [a + "_h" for a in df.index])
    vt = pd.DataFrame(np.zeros_like(df), columns = df.columns, index = [a + "_v" for a in df.index])
    rt = pd.DataFrame(np.zeros_like(df), columns = df.columns, index = [a + "_r" for a in df.index])

    hkey = ["theta" + str(i) for i in np.array([3, 2, 1, 0, 7, 6, 5, 4], dtype = int) + 1]
    vkey = ["theta" + str(i) for i in np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype = int) + 1]
    rkey = ["theta" + str(i) for i in np.array([4, 5, 6, 7, 0, 1, 2, 3], dtype = int) + 1]

    #swap drum positions
    ht.loc[:, hkey] = df.loc[:, ["theta" + str(i+1) for i in range(8)]].values
    vt.loc[:, vkey] = df.loc[:, ["theta" + str(i+1) for i in range(8)]].values
    rt.loc[:, rkey] = df.loc[:, ["theta" + str(i+1) for i in range(8)]].values

    #adjust angles
    ht.loc[:, hkey] = (3*np.pi - ht.loc[:, hkey]) % (2 * np.pi)
    vt.loc[:, vkey] = (2*np.pi - vt.loc[:, hkey]) % (2 * np.pi)
    rt.loc[:, rkey] = (np.pi + rt.loc[:, hkey]) % (2 * np.pi)

    #fill criticalities
    ht.iloc[:, [0, 1]] = df.iloc[:, [0,1]].values
    vt.iloc[:, [0, 1]] = df.iloc[:, [0,1]].values
    rt.iloc[:, [0, 1]] = df.iloc[:, [0,1]].values

    #combine dfs
    combined = pd.concat([df, ht, vt, rt])

    return combined

#def qpower_preprocess(df, sym = True):
#    if mult_sym
