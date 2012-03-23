

from spatial_model_generator import constructVAR, make_model_geofield,\
    constructVAR2
import numpy as np
import matplotlib.pyplot as plt
from geo_field import GeoField
from surr_geo_field_ar import SurrGeoFieldAR
import cPickle

# construct the testing model from a spec
S = np.zeros(shape = (20, 50), dtype = np.int32)
S[10:18, 25:45] = 1
S[0:3, 6:12] = 2
v, Sr = constructVAR(S, [0.0, 0.8, 0.8], [-0.1, 0.1], [0.0, 0.0])

#v, Sr = constructVAR2(S, [-0.2, 0.2], [0.0, 0.9, 0.9], 0.8)

#S = np.zeros(shape = (5, 10), dtype = np.int32)
#S[1:4, 0:2] = 1
#S[0:3, 6:9] = 2v, Sr = constructVAR(S, [0.0, 0.191, 0.120], [-0.1, 0.1], [0.00, 0.00], [0.01, 0.01])
ts = v.simulate(768)

gf = make_model_geofield(S, ts)
sgf = SurrGeoFieldAR()
sgf.copy_field(gf)
sgf.prepare_surrogates()
sgf.construct_surrogate_with_noise()
ts2 = sgf.surr_data() 

plt.figure(figsize = (8, 8))
plt.imshow(S, interpolation = 'nearest')
plt.title('Structural matrix')

plt.figure()
plt.imshow(v.A, interpolation = 'nearest')
plt.colorbar()
plt.title('AR structural')

plt.figure()
plt.plot(ts)
plt.title('Simulated time series')

C = np.corrcoef(ts, None, rowvar = 0)

plt.figure()
plt.imshow(C, interpolation = 'nearest')
plt.title('Correlation matrix')
plt.colorbar()

gf = GeoField()
gf.lons = np.arange(S.shape[1])
gf.lats = np.arange(S.shape[0])
gf.tm = np.arange(768)
gf.d = np.reshape(ts, [768, S.shape[0], S.shape[1]])

plt.figure()
plt.plot(ts2[:,0])
plt.plot(gf.d[:, 0, 0])

with open('data/test_gf.bin', 'w') as f:
    cPickle.dump(gf, f)

plt.show()
