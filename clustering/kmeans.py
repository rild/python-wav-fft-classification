import pickle

def loader(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

fname = 'hanekawa_nandemoha01'

rpath = 'pkls/'
t_filename = rpath + fname + '_t.pickle'
f_filename = rpath + fname + '_f.pickle'
Zxx_filename = rpath + fname + '_Zxx.pickle'

def stft_data_loader(ffile, tfile, Zxxfile):
    f = loader(ffile)
    t = loader(tfile)
    Zxx = loader(Zxxfile)
    return f, t, Zxx


f, t, Zxx = stft_data_loader(f_filename,
                             t_filename,
                             Zxx_filename)

print(f.shape)
print(t.shape)
print(Zxx.shape)

ZxxT = Zxx.T


from sklearn.cluster import KMeans
import numpy as np

kmeans = KMeans(n_clusters=300, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0).fit(ZxxT)
#  t_km = km.fit_predict(ZxxT)

labels = kmeans.labels_
# print(labels)
# print(kmeans.cluster_centers_[labels[0]])

first = True
for i in labels:
    if first:
        centers = kmeans.cluster_centers_[labels[0]]  # TO FIX
        first = False
    else:
        centers = np.vstack((centers, kmeans.cluster_centers_[i]))
print(centers.shape)

ZxxC = centers.T
print(ZxxC.shape)

with open(rpath + 'generated' + 'ZxxC.pickle', 'wb') as file:
    pickle.dump(ZxxC, file)

# with open('hanekawa_nandemoha01' + '_t_km.pickle', 'wb') as file:
#     pickle.dump(t_km, file)


'''

http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
Examples

'''