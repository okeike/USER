"""Evaluation"""
from __future__ import print_function
import logging
import time
import torch
import numpy as np

from collections import OrderedDict
from transformers import BertTokenizer

from lib.datasets import image_caption
from lib.vse import VSEModel_S2, VSEModel_S1
from lib.encoders import l1norm

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)

'''encode_data_dual & eval_rank_dual'''

def encode_data_dual(model, data_loader, log_step=10, logging=logger.info, backbone=False):
    """
    为DualVSEModel编码数据，分别返回VSE1和VSE2的嵌入
    init
    Args:
        model: DualVSEModel实例
        data_loader: 数据加载器
        log_step: 日志打印间隔
        logging: 日志函数
        backbone: 是否使用backbone模式

    Returns:
        img_embs: VSE的图像嵌入 (numpy array)
        cap_embs: VSE的文本嵌入 (numpy array)
        all_img_mlp: 所有batch的img_mlp (torch.Tensor, 3D)
        all_cap_lin: 所有batch的cap_lin (torch.Tensor, 3D)
        all_image_lengths: 合并后的image_lengths (numpy array, 1D)
        all_lengths: 合并后的lengths (numpy array, 1D)
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    # 收集每个batch的数据
    all_img_mlp = []
    all_cap_lin = []
    all_image_lengths = []
    all_lengths = []

    for i, data_i in enumerate(data_loader):
        # make sure val logger is used
        if not backbone:
            images, image_lengths, captions, lengths, ids, tag_index, img_ids, _, _ = data_i
        else:
            images, captions, lengths, ids, tag_index, img_ids = data_i
            image_lengths = None

        model.logger = val_logger

        # compute the embeddings from both VSE1 and VSE2
        if not backbone:
            img_emb_vse, cap_emb_vse, img_mlp, cap_lin = model.forward_emb(
                images, captions, lengths, image_lengths=image_lengths
            )
        else:
            img_emb_vse, cap_emb_vse, img_mlp, cap_lin = model.forward_emb(
                images, captions, lengths
            )

        # 收集当前batch的数据 - 保持为tensor
        if isinstance(img_mlp, torch.Tensor):
            all_img_mlp.append(img_mlp.cpu())
        else:
            all_img_mlp.append(torch.tensor(img_mlp))

        if isinstance(cap_lin, torch.Tensor):
            all_cap_lin.append(cap_lin.cpu())
        else:
            all_cap_lin.append(torch.tensor(cap_lin))

        # 处理image_lengths
        if image_lengths is not None:
            if isinstance(image_lengths, torch.Tensor):
                all_image_lengths.append(image_lengths.cpu().numpy().copy())
            elif isinstance(image_lengths, np.ndarray):
                all_image_lengths.append(image_lengths.copy())
            elif isinstance(image_lengths, list):
                all_image_lengths.append(np.array(image_lengths))
            else:
                all_image_lengths.append(np.array([image_lengths]))
        else:
            all_image_lengths.append(None)

        # 处理lengths
        if isinstance(lengths, torch.Tensor):
            all_lengths.append(lengths.cpu().numpy().copy())
        elif isinstance(lengths, np.ndarray):
            all_lengths.append(lengths.copy())
        elif isinstance(lengths, list):
            all_lengths.append(np.array(lengths))
        else:
            all_lengths.append(np.array([lengths]))

        # initialize the numpy arrays on first batch
        if img_embs is None:
            # Initialize VSE1 embeddings
            if img_emb_vse.dim() == 3:
                img_embs = np.zeros((len(data_loader.dataset), img_emb_vse.size(1), img_emb_vse.size(2)))
            else:
                img_embs = np.zeros((len(data_loader.dataset), img_emb_vse.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb_vse.size(1)))

        # cache VSE1 embeddings
        img_embs[ids] = img_emb_vse.data.cpu().numpy().copy()
        cap_embs[ids, :] = cap_emb_vse.data.cpu().numpy().copy()

        # measure accuracy and record loss
        model.forward_loss(img_emb_vse, cap_emb_vse)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            .format(
                i, len(data_loader.dataset) // data_loader.batch_size + 1,
                batch_time=batch_time,
                e_log=str(model.logger)))
        del images, captions

    # 根据最大长度进行padding并构造三维张量
    # 计算总样本数
    total_samples = sum(t.size(0) for t in all_img_mlp)

    # 获取特征维度
    img_feat_dim = all_img_mlp[0].size(-1) if all_img_mlp else 0
    cap_feat_dim = all_cap_lin[0].size(-1) if all_cap_lin else 0

    # 合并所有batch的lengths为一维数组
    # 处理image_lengths
    merged_image_lengths = []
    for img_lens in all_image_lengths:
        if img_lens is not None:
            if isinstance(img_lens, np.ndarray):
                merged_image_lengths.extend(img_lens.flatten().tolist())
            elif isinstance(img_lens, (list, tuple)):
                merged_image_lengths.extend(img_lens)
            else:
                merged_image_lengths.append(int(img_lens))

    # 处理cap_lengths
    merged_cap_lengths = []
    for cap_lens in all_lengths:
        if cap_lens is not None:
            if isinstance(cap_lens, np.ndarray):
                merged_cap_lengths.extend(cap_lens.flatten().tolist())
            elif isinstance(cap_lens, (list, tuple)):
                merged_cap_lengths.extend(cap_lens)
            else:
                merged_cap_lengths.append(int(cap_lens))

    # 转换为numpy数组
    merged_image_lengths = np.array(merged_image_lengths) if merged_image_lengths else np.array([])
    merged_cap_lengths = np.array(merged_cap_lengths) if merged_cap_lengths else np.array([])

    # 计算最大长度
    max_img_len = int(merged_image_lengths.max()) if len(merged_image_lengths) > 0 else 0
    max_cap_len = int(merged_cap_lengths.max()) if len(merged_cap_lengths) > 0 else 0

    # 如果没有长度信息，使用实际tensor的序列长度
    if max_img_len == 0 and all_img_mlp:
        max_img_len = max(t.size(1) for t in all_img_mlp if t.dim() >= 2)
    if max_cap_len == 0 and all_cap_lin:
        max_cap_len = max(t.size(1) for t in all_cap_lin if t.dim() >= 2)

    # 构造padding后的三维张量
    if all_img_mlp and max_img_len > 0:
        padded_img_mlp = torch.zeros(total_samples, max_img_len, img_feat_dim)
        idx = 0
        for img_mlp_batch in all_img_mlp:
            batch_size = img_mlp_batch.size(0)
            seq_len = img_mlp_batch.size(1) if img_mlp_batch.dim() >= 2 else 1
            if img_mlp_batch.dim() == 2:
                padded_img_mlp[idx:idx + batch_size, :seq_len, :] = img_mlp_batch
            elif img_mlp_batch.dim() == 3:
                padded_img_mlp[idx:idx + batch_size, :seq_len, :] = img_mlp_batch
            idx += batch_size
        all_img_mlp = padded_img_mlp
    else:
        all_img_mlp = torch.tensor([])

    if all_cap_lin and max_cap_len > 0:
        padded_cap_lin = torch.zeros(total_samples, max_cap_len, cap_feat_dim)
        idx = 0
        for cap_lin_batch in all_cap_lin:
            batch_size = cap_lin_batch.size(0)
            seq_len = cap_lin_batch.size(1) if cap_lin_batch.dim() >= 2 else 1
            if cap_lin_batch.dim() == 2:
                padded_cap_lin[idx:idx + batch_size, :seq_len, :] = cap_lin_batch
            elif cap_lin_batch.dim() == 3:
                padded_cap_lin[idx:idx + batch_size, :seq_len, :] = cap_lin_batch
            idx += batch_size
        all_cap_lin = padded_cap_lin
    else:
        all_cap_lin = torch.tensor([])

    return img_embs, cap_embs, all_img_mlp, all_cap_lin, merged_image_lengths, merged_cap_lengths

def evalrank_s1(model_path, data_path=None, split='dev', fold5=False, save_path=None):

    # load model and options
    checkpoint = torch.load(model_path, weights_only=False)
    opt = checkpoint['opt']
    opt.workers = 0

    logger.info(opt)
    if not hasattr(opt, 'caption_loss'):
        opt.caption_loss = False

    # load vocabulary used by the model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)

    opt.backbone_path = 'data/weights/original_updown_backbone.pth'
    if data_path is not None:
        opt.data_path = data_path

    # construct model
    model = VSEModel_S1(opt)
    logger.info("=> loading checkpoint from epoch '{}'".format(checkpoint['epoch']))
    # model.update_epoch(checkpoint['epoch'])
    model.make_data_parallel()
    # load model state
    model.load_state_dict(checkpoint['model'])
    model.val_start()

    logger.info('Loading dataset')
    data_loader = image_caption.get_test_loader(split, opt.data_name, tokenizer,
                                                opt.batch_size, opt.workers, opt)

    logger.info('Computing results...')
    with torch.no_grad():
        if opt.precomp_enc_type == 'basic':
            img_embs_vse, cap_embs_vse, img_mlp, cap_lin, img_len, cap_len = encode_data_dual(model, data_loader)
        else:
            img_embs_vse, cap_embs_vse, img_mlp, cap_lin, img_len, cap_len = encode_data_dual(model, data_loader, backbone=True)

    logger.info('Images: %d, Captions: %d' %
                (img_embs_vse.shape[0] / 5, cap_embs_vse.shape[0]))
    if not fold5:
        # ==================== 评估VSE1 ====================
        logger.info('\n' + '=' * 50)
        logger.info('VSE1 Performance:')
        logger.info('=' * 50)

        # no cross-validation, full evaluation
        img_embs_vse1_eval = np.array([img_embs_vse[i] for i in range(0, len(img_embs_vse), 5)])
        start = time.time()

        sims_vse1 = compute_sim(img_embs_vse1_eval, cap_embs_vse)
        print(sims_vse1.shape)
        npts = img_embs_vse1_eval.shape[0]
        if save_path is not None:
            np.save(save_path, {'npts': npts, 'sims': sims_vse1})
            logger.info('Save the similarity into {}'.format(save_path))
        end = time.time()
        logger.info("VSE1 - calculate similarity time: {}".format(end - start))

        r, rt = i2t(npts, sims_vse1, return_ranks=True)
        ri, rti = t2i(npts, sims_vse1, return_ranks=True)
        ar_vse1 = (r[0] + r[1] + r[2]) / 3
        ari_vse1 = (ri[0] + ri[1] + ri[2]) / 3
        rsum_vse1 = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        logger.info("VSE1 - rsum: %.1f" % rsum_vse1)
        logger.info("VSE1 - Average i2t Recall: %.1f" % ar_vse1)
        logger.info("VSE1 - Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        logger.info("VSE1 - Average t2i Recall: %.1f" % ari_vse1)
        logger.info("VSE1 - Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)

    else:
        # ==================== 5折交叉验证 ====================
        # 评估VSE1
        logger.info('\n' + '=' * 50)
        logger.info('VSE1 Performance (5-fold):')
        logger.info('=' * 50)

        results_vse1 = []
        for i in range(5):
            img_embs_shard = img_embs_vse[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs_vse[i * 5000:(i + 1) * 5000]
            start = time.time()
            sims_vse1 = compute_sim(img_embs_shard, cap_embs_shard)
            end = time.time()
            logger.info("VSE1 - Fold %d - calculate similarity time: {}".format(end - start) % i)

            npts = img_embs_shard.shape[0]
            r, rt0 = i2t(npts, sims_vse1, return_ranks=True)
            logger.info("VSE1 - Fold %d - Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                        (i, r[0], r[1], r[2], r[3], r[4]))
            ri, rti0 = t2i(npts, sims_vse1, return_ranks=True)
            logger.info("VSE1 - Fold %d - Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                        (i, ri[0], ri[1], ri[2], ri[3], ri[4]))

            if i == 0:
                rt_vse1, rti_vse1 = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            logger.info("VSE1 - Fold %d - rsum: %.1f ar: %.1f ari: %.1f" % (i, rsum, ar, ari))
            results_vse1 += [list(r) + list(ri) + [ar, ari, rsum]]

        logger.info("-----------------------------------")
        logger.info("VSE1 - Mean metrics: ")
        mean_metrics_vse1 = tuple(np.array(results_vse1).mean(axis=0).flatten())
        logger.info("VSE1 - rsum: %.1f" % (mean_metrics_vse1[12]))
        logger.info("VSE1 - Average i2t Recall: %.1f" % mean_metrics_vse1[10])
        logger.info("VSE1 - Image to text: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics_vse1[:5])
        logger.info("VSE1 - Average t2i Recall: %.1f" % mean_metrics_vse1[11])
        logger.info("VSE1 - Text to image: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics_vse1[5:10])

def encode_data(model, data_loader, log_step=10, logging=logger.info, backbone=False):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    for i, data_i in enumerate(data_loader):
        # make sure val logger is used
        if not backbone:
            images, image_lengths, captions, lengths, ids, tag_index, img_ids, img_tri, tri_lengths = data_i
        else:
            images, captions, lengths, ids, tag_index, img_ids, img_tri, tri_lengths = data_i
        model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb = model.forward_emb(images, captions, lengths, image_lengths=image_lengths, img_tri= img_tri, tri_lengths=tri_lengths)

        if img_embs is None:
            if img_emb.dim() == 3:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            else:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
            cap_lens = [0] * len(data_loader.dataset)

        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, :] = cap_emb.data.cpu().numpy().copy()

        # measure accuracy and record loss
        model.forward_loss(img_emb, cap_emb)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                i, len(data_loader.dataset) // data_loader.batch_size + 1, batch_time=batch_time,
                e_log=str(model.logger)))
        del images, captions
    return img_embs, cap_embs

def eval_ensemble(results_paths, fold5=False):
    all_sims = []
    all_npts = []
    for sim_path in results_paths:
        results = np.load(sim_path, allow_pickle=True).tolist()
        npts = results['npts']
        sims = results['sims']
        all_npts.append(npts)
        all_sims.append(sims)
    all_npts = np.array(all_npts)
    all_sims = np.array(all_sims)
    assert np.all(all_npts == all_npts[0])
    npts = int(all_npts[0])
    sims = all_sims.mean(axis=0)

    if not fold5:
        r, rt = i2t(npts, sims, return_ranks=True)
        ri, rti = t2i(npts, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        logger.info("rsum: %.1f" % rsum)
        logger.info("Average i2t Recall: %.1f" % ar)
        logger.info("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        logger.info("Average t2i Recall: %.1f" % ari)
        logger.info("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        npts = npts // 5
        results = []
        all_sims = sims.copy()
        for i in range(5):
            sims = all_sims[i * npts: (i + 1) * npts, i * npts * 5: (i + 1) * npts * 5]
            r, rt0 = i2t(npts, sims, return_ranks=True)
            logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(npts, sims, return_ranks=True)
            logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            logger.info("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]
        logger.info("-----------------------------------")
        logger.info("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        logger.info("rsum: %.1f" % (mean_metrics[12]))
        logger.info("Average i2t Recall: %.1f" % mean_metrics[10])
        logger.info("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics[:5])
        logger.info("Average t2i Recall: %.1f" % mean_metrics[11])
        logger.info("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics[5:10])


def evalrank_s2(model_path, data_path=None, split='dev', fold5=False, save_path=None):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path, weights_only=False)
    opt = checkpoint['opt']
    opt.workers = 5

    logger.info(opt)
    if not hasattr(opt, 'caption_loss'):
        opt.caption_loss = False

    # load vocabulary used by the model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)

    opt.backbone_path = 'data/weights/original_updown_backbone.pth'
    if data_path is not None:
        opt.data_path = data_path

    # construct model
    model = VSEModel_S2(opt)

    model.make_data_parallel()
    # load model state
    model.load_state_dict(checkpoint['model'])
    model.val_start()

    logger.info('Loading dataset')
    data_loader = image_caption.get_test_loader(split, opt.data_name, tokenizer,
                                                opt.batch_size, opt.workers, opt)

    logger.info('Computing results...')
    with torch.no_grad():
        if opt.precomp_enc_type == 'basic':
            img_embs, cap_embs = encode_data(model, data_loader)
        else:
            img_embs, cap_embs = encode_data(model, data_loader, backbone=True)
    logger.info('Images: %d, Captions: %d' %
                (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
        start = time.time()

        sims = compute_sim(img_embs, cap_embs)
        npts = img_embs.shape[0]

        if save_path is not None:
            np.save(save_path, {'npts': npts, 'sims': sims})
            logger.info('Save the similarity into {}'.format(save_path))

        end = time.time()
        logger.info("calculate similarity time: {}".format(end - start))

        r, rt = i2t(npts, sims, return_ranks=True)
        ri, rti = t2i(npts, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        logger.info("rsum: %.1f" % rsum)
        logger.info("Average i2t Recall: %.1f" % ar)
        logger.info("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        logger.info("Average t2i Recall: %.1f" % ari)
        logger.info("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            start = time.time()
            sims = compute_sim(img_embs_shard, cap_embs_shard)
            end = time.time()
            logger.info("calculate similarity time: {}".format(end - start))

            npts = img_embs_shard.shape[0]
            r, rt0 = i2t(npts, sims, return_ranks=True)
            logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(npts, sims, return_ranks=True)
            logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            logger.info("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        logger.info("-----------------------------------")
        logger.info("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        logger.info("rsum: %.1f" % (mean_metrics[12]))
        logger.info("Average i2t Recall: %.1f" % mean_metrics[10])
        logger.info("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics[:5])
        logger.info("Average t2i Recall: %.1f" % mean_metrics[11])
        logger.info("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics[5:10])


def compute_sim(images, captions):
    similarities = np.matmul(images, np.matrix.transpose(captions))
    return similarities


def compute_sim_new(img_embs, cap_embs, img_lens, cap_lens, model, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

    d = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            # 获取图像和文本嵌入切片
            im = img_embs[im_start:im_end]
            s = cap_embs[cap_start:cap_end]

            # 获取长度切片
            il = img_lens[im_start:im_end]
            sl = cap_lens[cap_start:cap_end]

            # 转换嵌入为tensor
            if isinstance(im, np.ndarray):
                im = torch.from_numpy(im).float().cuda()
            elif isinstance(im, torch.Tensor):
                im = im.float().cuda()

            if isinstance(s, np.ndarray):
                s = torch.from_numpy(s).float().cuda()
            elif isinstance(s, torch.Tensor):
                s = s.float().cuda()

            # 将长度转换为1D tensor
            if isinstance(il, np.ndarray):
                if il.ndim > 1:
                    # 取每个样本的实际长度（假设在第一列或需要取最大值）
                    il = torch.from_numpy(il[:, 0] if il.shape[1] > 0 else il.flatten()).cuda()
                else:
                    il = torch.from_numpy(il).cuda()
            elif isinstance(il, torch.Tensor):
                if il.dim() > 1:
                    il = il[:, 0].cuda() if il.size(1) > 0 else il.flatten().cuda()
                else:
                    il = il.cuda()
            elif isinstance(il, list):
                if il and isinstance(il[0], (list, np.ndarray)):
                    il = torch.tensor(
                        [item[0] if isinstance(item, (list, np.ndarray)) and len(item) > 0 else item for item in
                         il]).cuda()
                else:
                    il = torch.tensor(il).cuda()

            if isinstance(sl, np.ndarray):
                if sl.ndim > 1:
                    sl = torch.from_numpy(sl[:, 0] if sl.shape[1] > 0 else sl.flatten()).cuda()
                else:
                    sl = torch.from_numpy(sl).cuda()
            elif isinstance(sl, torch.Tensor):
                if sl.dim() > 1:
                    sl = sl[:, 0].cuda() if sl.size(1) > 0 else sl.flatten().cuda()
                else:
                    sl = sl.cuda()
            elif isinstance(sl, list):
                if sl and isinstance(sl[0], (list, np.ndarray)):
                    sl = torch.tensor(
                        [item[0] if isinstance(item, (list, np.ndarray)) and len(item) > 0 else item for item in
                         sl]).cuda()
                else:
                    sl = torch.tensor(sl).cuda()

            # 裁剪嵌入到实际需要的最大长度
            actual_max_r = int(il.max())
            actual_max_w = int(sl.max())

            # 确保嵌入的序列长度至少等于实际最大长度
            if im.size(1) > actual_max_r:
                im = im[:, :actual_max_r, :]
            if s.size(1) > actual_max_w:
                s = s[:, :actual_max_w, :]

            sim = model.forward_sim(im, s, il, sl)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()

    return d


def compute_sim_ours(img_embs, cap_embs, img_lens, cap_lens, model, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    使用基于差值的相似度计算方法（向量化优化版本）
    以步长5遍历图像
    """
    # 计算实际的图像样本数（每5个caption对应1个image）
    n_imgs = len(img_embs) // 5
    n_im_shard = (n_imgs - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

    d = np.zeros((n_imgs, len(cap_embs)))

    for i in range(n_im_shard):
        im_start_idx, im_end_idx = shard_size * i, min(shard_size * (i + 1), n_imgs)
        # 转换为实际的img_embs索引（步长为5）
        im_start, im_end = im_start_idx * 5, im_end_idx * 5

        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            # 获取图像和文本嵌入切片（图像以步长5采样）
            im = img_embs[im_start:im_end:5]  # 每5个取1个
            s = cap_embs[cap_start:cap_end]

            # 获取长度切片
            il = img_lens[im_start:im_end:5]  # 每5个取1个
            sl = cap_lens[cap_start:cap_end]

            # 转换嵌入为tensor
            if isinstance(im, np.ndarray):
                im = torch.from_numpy(im).float().cuda()
            elif isinstance(im, torch.Tensor):
                im = im.float().cuda()

            if isinstance(s, np.ndarray):
                s = torch.from_numpy(s).float().cuda()
            elif isinstance(s, torch.Tensor):
                s = s.float().cuda()

            # 将长度转换为1D tensor
            if isinstance(il, np.ndarray):
                if il.ndim > 1:
                    il = torch.from_numpy(il[:, 0] if il.shape[1] > 0 else il.flatten()).cuda()
                else:
                    il = torch.from_numpy(il).cuda()
            elif isinstance(il, torch.Tensor):
                if il.dim() > 1:
                    il = il[:, 0].cuda() if il.size(1) > 0 else il.flatten().cuda()
                else:
                    il = il.cuda()
            elif isinstance(il, list):
                if il and isinstance(il[0], (list, np.ndarray)):
                    il = torch.tensor(
                        [item[0] if isinstance(item, (list, np.ndarray)) and len(item) > 0 else item for item in il]
                    ).cuda()
                else:
                    il = torch.tensor(il).cuda()

            if isinstance(sl, np.ndarray):
                if sl.ndim > 1:
                    sl = torch.from_numpy(sl[:, 0] if sl.shape[1] > 0 else sl.flatten()).cuda()
                else:
                    sl = torch.from_numpy(sl).cuda()
            elif isinstance(sl, torch.Tensor):
                if sl.dim() > 1:
                    sl = sl[:, 0].cuda() if sl.size(1) > 0 else sl.flatten().cuda()
                else:
                    sl = sl.cuda()
            elif isinstance(sl, list):
                if sl and isinstance(sl[0], (list, np.ndarray)):
                    sl = torch.tensor(
                        [item[0] if isinstance(item, (list, np.ndarray)) and len(item) > 0 else item for item in sl]
                    ).cuda()
                else:
                    sl = torch.tensor(sl).cuda()

            # 裁剪嵌入到实际需要的最大长度
            actual_max_r = int(il.max())
            actual_max_w = int(sl.max())

            # 确保嵌入的序列长度至少等于实际最大长度
            if im.size(1) > actual_max_r:
                im = im[:, :actual_max_r, :]
            if s.size(1) > actual_max_w:
                s = s[:, :actual_max_w, :]

            # 向量化的相似度计算
            # im: [batch_i, max_img_len, feat_dim]
            # s: [batch_c, max_cap_len, feat_dim]
            batch_i = im.size(0)
            batch_c = s.size(0)
            feat_dim = im.size(2)

            # L1归一化
            im_norm = l1norm(im, dim=-1)  # [batch_i, max_img_len, feat_dim]
            s_norm = l1norm(s, dim=-1)  # [batch_c, max_cap_len, feat_dim]

            # 创建mask
            img_mask = torch.arange(actual_max_r, device=im.device).unsqueeze(0) < il.unsqueeze(1)
            cap_mask = torch.arange(actual_max_w, device=s.device).unsqueeze(0) < sl.unsqueeze(1)

            # 重塑用于批量计算
            # im_norm: [batch_i, max_img_len, feat_dim] -> [batch_i, 1, max_img_len, feat_dim]
            # s_norm: [batch_c, max_cap_len, feat_dim] -> [1, batch_c, max_cap_len, feat_dim]
            im_norm_exp = im_norm.unsqueeze(1)  # [batch_i, 1, max_img_len, feat_dim]
            s_norm_exp = s_norm.unsqueeze(0)  # [1, batch_c, max_cap_len, feat_dim]

            # 扩展到相同维度
            im_norm_exp = im_norm_exp.expand(batch_i, batch_c, actual_max_r,
                                             feat_dim)  # [batch_i, batch_c, max_img_len, feat_dim]
            s_norm_exp = s_norm_exp.expand(batch_i, batch_c, actual_max_w,
                                           feat_dim)  # [batch_i, batch_c, max_cap_len, feat_dim]

            # 计算余弦相似度
            # s_norm_exp: [batch_i, batch_c, max_cap_len, feat_dim]
            # im_norm_exp: [batch_i, batch_c, max_img_len, feat_dim]
            # 结果: [batch_i, batch_c, max_cap_len, max_img_len]
            cos_sim = torch.matmul(
                s_norm_exp,  # [batch_i, batch_c, max_cap_len, feat_dim]
                im_norm_exp.transpose(-1, -2)  # [batch_i, batch_c, feat_dim, max_img_len]
            )  # -> [batch_i, batch_c, max_cap_len, max_img_len]

            # 应用image mask（无效位置设为-inf）
            img_mask_exp = img_mask.unsqueeze(1).unsqueeze(2)  # [batch_i, 1, 1, max_img_len]
            cos_sim = cos_sim.masked_fill(~img_mask_exp, -1e9)

            # 对每个caption向量，找到最相似的image向量索引
            # [batch_i, batch_c, max_cap_len]
            max_indices = torch.argmax(cos_sim, dim=-1)

            # 使用gather获取最相似的image向量
            # max_indices: [batch_i, batch_c, max_cap_len, 1]
            max_indices_exp = max_indices.unsqueeze(-1).expand(-1, -1, -1, feat_dim)

            # 收集最相似的image向量
            # [batch_i, batch_c, max_cap_len, feat_dim]
            max_img_vecs = torch.gather(im_norm_exp, 2, max_indices_exp)

            # 计算差值
            # s_norm_exp: [batch_i, batch_c, max_cap_len, feat_dim]
            # max_img_vecs: [batch_i, batch_c, max_cap_len, feat_dim]
            diff = s_norm_exp - max_img_vecs  # [batch_i, batch_c, max_cap_len, feat_dim]

            # L1归一化差值
            diff_norm = l1norm(diff, dim=-1)  # [batch_i, batch_c, max_cap_len, feat_dim]

            # 计算绝对值和
            abs_sum = torch.abs(diff_norm).sum(dim=-1)  # [batch_i, batch_c, max_cap_len]

            # 应用caption mask并求平均
            cap_mask_exp = cap_mask.unsqueeze(0)  # [1, batch_c, max_cap_len]
            abs_sum = abs_sum.masked_fill(~cap_mask_exp, 0)

            # 计算每个样本的有效caption长度
            valid_cap_lens = cap_mask_exp.sum(dim=-1).float()  # [1, batch_c]

            # 求平均（避免除零）
            avg_diff = abs_sum.sum(dim=-1) / (valid_cap_lens + 1e-8)  # [batch_i, batch_c]

            # 转换为相似度（取倒数）
            similarities = 1.0 / (avg_diff + 1e-8)

            # 转换为numpy并存储
            d[im_start_idx:im_end_idx, cap_start:cap_end] = similarities.cpu().numpy()

    return d

def i2t(npts, sims, return_ranks=False, mode='coco'):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        if mode == 'coco':
            rank = 1e20
            for i in range(5 * index, 5 * index + 5, 1):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
            top1[index] = inds[0]
        else:
            rank = np.where(inds == index)[0][0]
            ranks[index] = rank
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(npts, sims, return_ranks=False, mode='coco'):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    # npts = images.shape[0]

    if mode == 'coco':
        ranks = np.zeros(5 * npts)
        top1 = np.zeros(5 * npts)
    else:
        ranks = np.zeros(npts)
        top1 = np.zeros(npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        if mode == 'coco':
            for i in range(5):
                inds = np.argsort(sims[5 * index + i])[::-1]
                ranks[5 * index + i] = np.where(inds == index)[0][0]
                top1[5 * index + i] = inds[0]
        else:
            inds = np.argsort(sims[index])[::-1]
            ranks[index] = np.where(inds == index)[0][0]
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
