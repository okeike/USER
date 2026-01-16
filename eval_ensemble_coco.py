import numpy as np
import time

from lib.evaluation import i2t, t2i

sim_file1 = 'runs/coco_1st_butd_region_bert/results_coco.npy'
sim_file2 = 'runs/coco_2nd_butd_region_bert/results_coco.npy'

print('Loading similarity matrices...')
data1 = np.load(sim_file1, allow_pickle=True).item()
data2 = np.load(sim_file2, allow_pickle=True).item()

sims_vse1 = data1['sims']
sims_vse2 = data2['sims']
npts = data1['npts']

print('Similarity matrix 1 shape: {}'.format(sims_vse1.shape))
print('Similarity matrix 2 shape: {}'.format(sims_vse2.shape))
print('Number of points: {}'.format(npts))

start = time.time()
sims_combined = sims_vse1 + sims_vse2

print('Combined similarity matrices')

save_path_combined = None
if save_path_combined is not None:
    np.save(save_path_combined, {'npts': npts, 'sims': sims_combined})
    print('Save the combined similarity into {}'.format(save_path_combined))

end = time.time()
print("Calculate combined similarity time: {}".format(end - start))

# ==================== Full evaluation (no cross-validation) ====================
print("\n" + "=" * 50)
print("Full Evaluation (no cross-validation)")
print("=" * 50)

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

# ==================== 5fold cross-validation ====================
print("\n" + "=" * 50)
print("5-fold Cross-validation")
print("=" * 50)

results = []
for i in range(5):
    # sims形状是(5000, 25000)，即5000张图片对25000个caption
    # 每个fold: 取1000张图片(第i个1000)，对应5000个caption(第i个5000)
    sims_shard = sims_combined[i * 1000:(i + 1) * 1000, i * 5000:(i + 1) * 5000]

    start = time.time()
    end = time.time()
    print("Fold %d - calculate similarity time: {}".format(i, end - start))

    npts_shard = sims_shard.shape[0]
    r, rt0 = i2t(npts_shard, sims_shard, return_ranks=True)
    print("Fold %d - Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (i, r[0], r[1], r[2], r[3], r[4]))

    ri, rti0 = t2i(npts_shard, sims_shard, return_ranks=True)
    print("Fold %d - Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (i, ri[0], ri[1], ri[2], ri[3], ri[4]))

    if i == 0:
        rt, rti = rt0, rti0

    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("Fold %d - rsum: %.1f ar: %.1f ari: %.1f" % (i, rsum, ar, ari))
    results += [list(r) + list(ri) + [ar, ari, rsum]]

print("-----------------------------------")
print("Mean metrics: ")
mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
print("rsum: %.1f" % (mean_metrics[12]))
print("Average i2t Recall: %.1f" % mean_metrics[10])
print("Image to text: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[:5])
print("Average t2i Recall: %.1f" % mean_metrics[11])
print("Text to image: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[5:10])