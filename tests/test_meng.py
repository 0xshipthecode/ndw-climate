
import numpy as np
import matplotlib.pyplot as plt

import spca_meng
import component_analysis


def make_test_data1(N, sigma):
    
    # generate source variables (np.random.normal scale parameter == standard deviation)
    V = np.zeros(shape = (N,3))
    V[:,0] = np.random.normal(size = (N,), scale = 290)
    V[:,1] = np.random.normal(size = (N,),scale = 300)
    V[:,2] = -0.3 * V[:,0] + 0.925 * V[:,1] + np.random.normal(size = (N,))
    
    # generate N 10D vectors
    X = np.zeros(shape=(N,10))
    v_sel = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    for i in range(10):
        X[:,i] = V[:,v_sel[i]] + np.random.normal(size = (N,), scale = sigma)
    
    return X
    

if __name__ == '__main__':
    
    # program the prolific test example used by everybody
    
    X = make_test_data1(1000, 1.0)
    
    C = spca_meng.extract_sparse_components(X.T, 3, 3)
    PC,_,_ = component_analysis.pca_components(X.T)
    PC = PC[:,:3]
    UR, _, _ = component_analysis.orthomax(PC)
    
    print C
    
    plt.figure()
    plt.subplot(311)
    plt.plot(C)
    plt.title('Sparse PCA')
    plt.subplot(312)
    plt.plot(PC)
    plt.title('PCA components')
    plt.subplot(313)
    plt.plot(UR)
    plt.title('Rotated components')
    plt.show()