import numpy as np
import pandas as pd

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
    thetas.loc[:, ["theta"+str(i) for i in [3,4,7,8]]] *= -1
    thetas.loc[:, thkeys] = \
            thetas.loc[:, thkeys] % (2*np.pi) #scale all to [0, 2*np.pi]
    for col in thkeys:
        thetas.loc[:, col].loc[thetas.loc[:, col] > np.pi] -= 2*np.pi #bring to [-pi, pi]
    return thetas

def l2g(thetas):
    """
    convert local coordinate system to the global one
    """
    loc_offsets = np.array([3.6820187359906447, 4.067668586955522,
                           2.2155167202240653-np.pi, 2.6011665711889425-np.pi,
                           0.5404260824008517, 0.9260759333657285,
                           5.3571093738138575-np.pi, 5.742759224778734-np.pi])
    thkeys = ["theta%i"%(i+1) for i in range(8)]
    thetas.loc[:, ["theta"+str(i) for i in [3,4,7,8]]] *= -1 #make ccw positive
    thetas.loc[:, thkeys] += loc_offsets + 2*np.pi
    thetas.loc[:, thkeys] = \
            thetas.loc[:, thkeys] % (2*np.pi) #scale all to [0, 2*np.pi]
    return thetas

if __name__ == "__main__":
    t = np.zeros((1, 8))
    thkeys = ["theta%i"%(i+1) for i in range(8)]
    tg = pd.DataFrame(t, columns = thkeys)
    print(l2g(tg)*180/np.pi)


