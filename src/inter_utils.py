

import cPickle


def load_results(fname):
    """Load a results file (usually extension .bin) which is a cPickle dump of the results."""
    with open(fname, 'r') as f:
        data = cPickle.load(f)

    return data

    
