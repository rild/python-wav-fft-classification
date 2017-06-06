import pickle

def loader(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

fname = 'hanekawa_nandemoha01'
t_filename = fname + '_t.pickle'
f_filename = fname + '_f.pickle'
Zxx_filename = fname + '_Zxx.pickle'

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

km = KMeans(n_clusters=300, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
t_km = km.fit_predict(ZxxT)

with open('hanekawa_nandemoha01' + '_t_km.pickle', 'wb') as file:
    pickle.dump(t_km, file)
