import numpy as np
import time

from lib.evaluation import i2t, t2i

sim_file1 = 'runs/f30k_vse1_butd_region_bert/results_f30k.npy'
sim_file2 = 'runs/f30k_vse2_butd_region_bert/results_f30k.npy'

print('Loading similarity matrices...')
data1 = np.load(sim_file1, allow_pickle=True).item()
data2 = np.load(sim_file2, allow_pickle=True).item()

sims_vse1 = data1['sims']
sims_vse2 = data2['sims']
npts = data1['npts']

print('Similarity matrix 1 shape: {}'.format(sims_vse1.shape))
print('Similarity matrix 2 shape: {}'.format(sims_vse2.shape))
print('Number of points: {}'.format(npts))

# 相加两个sim
start = time.time()
sims_combined = sims_vse1 + sims_vse2
print('Combined similarity matrices')

# 可选：保存合并后的sims
save_path_combined = None
if save_path_combined is not None:
    np.save(save_path_combined, {'npts': npts, 'sims': sims_combined})
    print('Save the combined similarity into {}'.format(save_path_combined))

end = time.time()
print("Calculate combined similarity time: {}".format(end - start))

r, rt = i2t(npts, sims_vse1, return_ranks=True)
ri, rti = t2i(npts, sims_vse1, return_ranks=True)
ar_combined = (r[0] + r[1] + r[2]) / 3
ari_combined = (ri[0] + ri[1] + ri[2]) / 3
rsum_combined = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]

print("sims_vse1 - rsum: %.1f" % rsum_combined)
print("sims_vse1 - Average i2t Recall: %.1f" % ar_combined)
print("sims_vse1 - Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
print("sims_vse1 - Average t2i Recall: %.1f" % ari_combined)
print("sims_vse1 - Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)

r, rt = i2t(npts, sims_vse2, return_ranks=True)
ri, rti = t2i(npts, sims_vse2, return_ranks=True)
ar_combined = (r[0] + r[1] + r[2]) / 3
ari_combined = (ri[0] + ri[1] + ri[2]) / 3
rsum_combined = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]

print("sims_vse2 - rsum: %.1f" % rsum_combined)
print("sims_vse2 - Average i2t Recall: %.1f" % ar_combined)
print("sims_vse2 - Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
print("sims_vse2 - Average t2i Recall: %.1f" % ari_combined)
print("sims_vse2 - Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)

r, rt = i2t(npts, sims_combined, return_ranks=True)
ri, rti = t2i(npts, sims_combined, return_ranks=True)
ar_combined = (r[0] + r[1] + r[2]) / 3
ari_combined = (ri[0] + ri[1] + ri[2]) / 3
rsum_combined = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]

print("Combined - rsum: %.1f" % rsum_combined)
print("Combined - Average i2t Recall: %.1f" % ar_combined)
print("Combined - Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
print("Combined - Average t2i Recall: %.1f" % ari_combined)
print("Combined - Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)