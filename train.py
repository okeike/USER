"""Training script"""
import os
import time
import numpy as np
import torch
from transformers import BertTokenizer

from lib.datasets import image_caption
from lib.vse import VSEModel_S1, VSEModel_S2
from lib.evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, encode_data_dual, compute_sim

import logging
import tensorboard_logger as tb_logger

import arguments


def main():
    # Hyper Parameters
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()

    if not os.path.exists(opt.model_name):
        os.makedirs(opt.model_name)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    logger = logging.getLogger(__name__)
    logger.info(opt)

    # Load Tokenizer and Vocabulary
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)

    train_loader, val_loader = image_caption.get_loaders(
        opt.data_path, opt.data_name, tokenizer, opt.batch_size, opt.workers, opt)

    if opt.model == 'VSE_S2':
        model = VSEModel_S2(opt)
    elif opt.model == 'VSE_S1':
        model = VSEModel_S1(opt)
    model.logger = tb_logger

    lr_schedules = [opt.lr_update, ]

    start_epoch = 0
    best_rsum = 0

    if opt.resume:
        if os.path.isfile(opt.resume):
            logger.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume, weights_only=False)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            if not model.is_data_parallel:
                model.make_data_parallel()
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another training
            model.Eiters = checkpoint['Eiters']

            logger.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                        .format(opt.resume, start_epoch, best_rsum))

            if opt.reset_start_epoch:
                start_epoch = 0
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.resume))

    if not model.is_data_parallel:
        model.make_data_parallel()

    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        logger.info(opt.logger_name)
        logger.info(opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch, lr_schedules)

        if epoch >= opt.vse_mean_warmup_epochs:
            opt.max_violation = True
            model.set_max_violation(opt.max_violation)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        if opt.model == 'VSE_S2':
            rsum = validate_s2(opt, val_loader, model)
        elif opt.model == 'VSE_S1':
            rsum = validate_s1(opt, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)

        if is_best:
            if not os.path.exists(opt.model_name):
                os.mkdir(opt.model_name)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, filename='checkpoint.pth', prefix=opt.model_name + '/')


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    if opt.model == 'VSE_S2':
        logger.info('image encoder trainable parameters: {}'.format(count_params(model.img_enc)))
        logger.info('txt encoder trainable parameters: {}'.format(count_params(model.txt_enc)))
    elif opt.model == 'VSE_S1':
        logger.info('first image encoder trainable parameters: {}'.format(count_params(model.img_enc)))
        logger.info('first txt encoder trainable parameters: {}'.format(count_params(model.txt_enc)))

    end = time.time()
    # opt.viz = True
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        images, img_lengths, captions, lengths, cap_idx, tag_index, img_ids, img_tri, tri_lengths = train_data
        if opt.model == 'VSE_S2':
            model.train_emb(images, captions, lengths, image_lengths=img_lengths, img_tri=img_tri, tri_lengths=tri_lengths)
        elif opt.model == 'VSE_S1':
            model.train_emb(images, captions, lengths, image_lengths=img_lengths, tag_index=tag_index, cap_idx=cap_idx)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger.info log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader.dataset) // train_loader.batch_size + 1, batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

def validate_s1(opt, val_loader, model):
    logger = logging.getLogger(__name__)
    model.val_start()

    with torch.no_grad():
        # compute the encoding for all the validation images and captions
        img_embs_vse, cap_embs_vse, img_mlp, cap_lin, img_len, cap_len = encode_data_dual(
            model, val_loader, opt.log_step, logging.info,
            backbone=opt.precomp_enc_type == 'backbone')

    # ==================== 评估VSE1 ====================
    logger.info('\n' + '=' * 50)
    logger.info('VSE1 Validation Performance:')
    logger.info('=' * 50)

    img_embs_vse1_eval = np.array([img_embs_vse[i] for i in range(0, len(img_embs_vse), 5)])

    start = time.time()
    sims_vse1 = compute_sim(img_embs_vse1_eval, cap_embs_vse)
    end = time.time()
    logger.info("VSE1 - calculate similarity time: {}".format(end - start))

    # caption retrieval
    npts = img_embs_vse1_eval.shape[0]
    (r1, r5, r10, medr, meanr) = i2t(npts, sims_vse1)
    logging.info("VSE1 - Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr_i) = t2i(npts, sims_vse1)
    logging.info("VSE1 - Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr_i))
    # sum of recalls to be used for early stopping
    currscore_vse1 = r1 + r5 + r10 + r1i + r5i + r10i
    logger.info('VSE1 - Current rsum is {}'.format(currscore_vse1))

    # record VSE1 metrics in tensorboard
    tb_logger.log_value('vse1_r1', r1, step=model.Eiters)
    tb_logger.log_value('vse1_r5', r5, step=model.Eiters)
    tb_logger.log_value('vse1_r10', r10, step=model.Eiters)
    tb_logger.log_value('vse1_medr', medr, step=model.Eiters)
    tb_logger.log_value('vse1_meanr', meanr, step=model.Eiters)
    tb_logger.log_value('vse1_r1i', r1i, step=model.Eiters)
    tb_logger.log_value('vse1_r5i', r5i, step=model.Eiters)
    tb_logger.log_value('vse1_r10i', r10i, step=model.Eiters)
    tb_logger.log_value('vse1_medri', medri, step=model.Eiters)
    tb_logger.log_value('vse1_meanr_i', meanr_i, step=model.Eiters)
    tb_logger.log_value('vse1_rsum', currscore_vse1, step=model.Eiters)

    return currscore_vse1

def validate_s2(opt, val_loader, model):
    logger = logging.getLogger(__name__)
    model.val_start()
    with torch.no_grad():
        # compute the encoding for all the validation images and captions
        img_embs, cap_embs = encode_data(
            model, val_loader, opt.log_step, logging.info, backbone=opt.precomp_enc_type == 'backbone')

    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    start = time.time()
    sims = compute_sim(img_embs, cap_embs)
    end = time.time()
    logger.info("VSE2 calculate similarity time: {}".format(end - start))

    # caption retrieval
    npts = img_embs.shape[0]
    (r1, r5, r10, medr, meanr) = i2t(npts, sims)
    logging.info("VSE2 Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))

    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(npts, sims)
    logging.info("VSE2 Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))

    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    logger.info('VSE2 Current rsum is {}'.format(currscore))

    # record metrics in tensorboard
    tb_logger.log_value('vse2_r1', r1, step=model.Eiters)
    tb_logger.log_value('vse2_r5', r5, step=model.Eiters)
    tb_logger.log_value('vse2_r10', r10, step=model.Eiters)
    tb_logger.log_value('vse2_medr', medr, step=model.Eiters)
    tb_logger.log_value('vse2_meanr', meanr, step=model.Eiters)
    tb_logger.log_value('vse2_r1i', r1i, step=model.Eiters)
    tb_logger.log_value('vse2_r5i', r5i, step=model.Eiters)
    tb_logger.log_value('vse2_r10i', r10i, step=model.Eiters)
    tb_logger.log_value('vse2_medri', medri, step=model.Eiters)
    tb_logger.log_value('vse2_meanr', meanr, step=model.Eiters)
    tb_logger.log_value('vse2_rsum', currscore, step=model.Eiters)

    return currscore

def save_checkpoint(state, is_best, filename='checkpoint.pth', prefix=''):
    logger = logging.getLogger(__name__)
    tries = 15

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix + 'model_best.pth')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        logger.info('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch, lr_schedules):
    logger = logging.getLogger(__name__)
    """Sets the learning rate to the initial LR
       decayed by 10 every opt.lr_update epochs"""
    if epoch in lr_schedules:
        logger.info('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * 0.1
            param_group['lr'] = new_lr
            logger.info('new lr {}'.format(new_lr))


def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


if __name__ == '__main__':
    main()
