
import matplotlib
matplotlib.use('agg')

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cPickle



if __name__ == '__main__':

    if len(sys.argv) < 4:
        print("Usage: corrmat_components.py <dataset1> <dataset2> <output-file-no-ext>")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        d1 = cPickle.load(f)

    with open(sys.argv[2], 'r') as f:
        d2 = cPickle.load(f)

    output_file = sys.argv[3]

    # retrieve data from results
    ts1, ts2 = np.asarray(d1['ts']), np.asarray(d2['ts'])
    N1, N2 = ts1.shape[0], ts2.shape[0]

    # compute the correlation coefficient
    C = np.corrcoef(ts1, ts2)
    C = C[0:N1, N1:N1+N2]
    C_all = C.copy()
    C[(C > -0.2) * (C < 0.2)] = 0.0
    # if its the same data, set diagonal to zero
    if sys.argv[1] == sys.argv[2]:
        C -= np.eye(N1)
    plt.figure(figsize = (16*(N2/N1), 16))
    plt.imshow(C, interpolation = 'nearest')
    for i in range(0,N1,5):
        plt.plot([-0.5, N2], [i-0.5, i-0.5], 'k-')
    for i in range(0,N2,5):
        plt.plot([i-0.5, i-0.5], [-0.5, N1], 'k-')
    plt.xlim(-0.5, N2-0.5)
    plt.ylim(-0.5, N1-0.5)
    plt.xticks(np.arange(0, N2, 2), np.arange(0, N2, 2)+1)
    plt.yticks(np.arange(0, N1, 2), np.arange(0, N1, 2)+1)
    plt.colorbar()
    plt.savefig(output_file + ".png")

    # store corr coef as file
    np.savetxt(output_file + ".csv", C_all, fmt = '%g', delimiter=";")

