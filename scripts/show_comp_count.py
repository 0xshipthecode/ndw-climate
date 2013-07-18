#!/usr/bin/env python

import numpy as np
import cPickle
import multi_stats
import sys


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: show_comp_count.py <comp_count_file> [Nhyp]")
        sys.exit(1)

    print("Loading file %s ..." % sys.argv[1])
    with open(sys.argv[1], 'r') as f:
        data = cPickle.load(f)

    Nhyp = int(sys.argv[2]) if len(sys.argv) >= 3 else None

    # load data and surrogate eigenvalues
    dlam = data['dlam']
    slam = data['slam_ar']

    p_vals = multi_stats.compute_eigvals_pvalues(dlam, slam)
    Nsurr = slam.shape[0]
    print("Bonferroni (single-level): %d components." % np.sum(multi_stats.bonferroni_test(p_vals, 0.05, Nsurr, Nhyp)))
    print("Sidak (single-level): %d components." % np.sum(multi_stats.sidak_test(p_vals, 0.05, Nsurr, Nhyp)))
    print("Bonferroni-Holm (stepdown): %d components." % np.sum(multi_stats.holm_test(p_vals, 0.05, Nsurr, Nhyp)))
    print("False Discovery Rate: %d components." % np.sum(multi_stats.fdr_test(p_vals, 0.05, Nsurr, Nhyp)))
