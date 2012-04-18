
import numpy as np
import mdp.nodes



if __name__ == '__main__':
    
    f = mdp.nodes.FastICANode()
    data = np.random.uniform(size=(100, 10))
    print(data.shape)
    r = f.execute(data)
    print r.shape