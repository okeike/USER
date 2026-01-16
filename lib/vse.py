"""VSE model"""

import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from lib.encoders import get_image_encoder, get_text_encoder, l1norm, l2norm
from lib.loss import ContrastiveLoss
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)

class VSEModel_S2(object):
    """
    The standard VSE model
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = get_image_encoder(opt.img_dim, opt.embed_size, no_imgnorm=opt.no_imgnorm, opt=opt)
        self.txt_enc = get_text_encoder(opt.embed_size, no_txtnorm=opt.no_txtnorm)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())

        self.params = params
        self.opt = opt

        # Set up the lr for different parts of the VSE model
        decay_factor = 1e-4
        if self.opt.optim == 'adam':
            all_text_params = list(self.txt_enc.parameters())
            bert_params = list(self.txt_enc.bert.parameters())
            bert_params_ptr = [p.data_ptr() for p in bert_params]
            text_params_no_bert = list()
            for p in all_text_params:
                if p.data_ptr() not in bert_params_ptr:
                    text_params_no_bert.append(p)
            self.optimizer = torch.optim.AdamW([
                {'params': text_params_no_bert, 'lr': opt.learning_rate},
                {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                {'params': self.img_enc.parameters(), 'lr': opt.learning_rate},
            ],
                lr=opt.learning_rate, weight_decay=decay_factor)
        elif self.opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.params, lr=opt.learning_rate, momentum=0.9)
        else:
            raise ValueError('Invalid optim option {}'.format(self.opt.optim))

        logger.info('Use {} as the optimizer, with init lr {}'.format(self.opt.optim, opt.learning_rate))

        self.Eiters = 0
        self.data_parallel = False

    def set_max_violation(self, max_violation):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.txt_enc.load_state_dict(state_dict[1], strict=False)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.data_parallel = True
        logger.info('Image encoder is data paralleled now.')

    @property
    def is_data_parallel(self):
        return self.data_parallel

    def forward_emb(self, images, captions, lengths, image_lengths=None, img_tri=None, tri_lengths=None):
        """
        Compute the image and caption embeddings

        Args:
            images: (batch_size, num_regions, img_dim)
            captions: (batch_size, max_cap_len)
            lengths: list or tensor
            image_lengths: (batch_size,)
            img_tri: (batch_size, max_num_tri, max_tri_len)
            tri_lengths: list of list，tri_lengths[i][j]: length of j triplets for ith sample

        Returns:
            img_emb: (batch_size, embed_size)
            cap_emb: (batch_size, embed_size)
        """
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            image_lengths = image_lengths.cuda()
            img_tri = img_tri.cuda()
        img_emb, _ = self.img_enc(images, image_lengths, img_tri, tri_lengths)
        lengths = torch.Tensor(lengths).cuda()
        cap_emb, _ = self.txt_enc(captions, lengths)
        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb)
        self.logger.update('Le', loss.data.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, image_lengths=None, img_tri=None, tri_lengths=None,
                  warmup_alpha=None):
        """
        One training step given images and captions.

        Args:
            images: (batch_size, num_regions, img_dim) 图像区域特征
            captions: (batch_size, max_cap_len) 文本caption的词索引
            lengths: (batch_size,) 每个caption的有效长度
            image_lengths: (batch_size,) 每个图像的有效区域数
            img_tri: (batch_size, max_num_tri, max_tri_len) 三元组词索引，填充值为0
            tri_lengths: list of list，tri_lengths[i][j]是第i个样本第j个三元组的长度
            warmup_alpha: warmup系数，用于调整loss
        """

        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        img_emb, cap_emb = self.forward_emb(
            images,
            captions,
            lengths,
            image_lengths=image_lengths,
            img_tri=img_tri,
            tri_lengths=tri_lengths
        )
        self.optimizer.zero_grad()

        loss = self.forward_loss(img_emb, cap_emb)

        if warmup_alpha is not None:
            loss = loss * warmup_alpha
        loss.backward()

        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)

        self.optimizer.step()

class VSEModel_S1(object):

    def __init__(self, opt):
        self.opt = opt
        self.grad_clip = opt.grad_clip

        self.img_enc = get_image_encoder(opt.img_dim, opt.embed_size, no_imgnorm=opt.no_imgnorm, opt=opt)
        self.txt_enc = get_text_encoder(opt.embed_size, no_txtnorm=opt.no_txtnorm)

        # Move to CUDA
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            torch.backends.cudnn.benchmark = True

        self.criterion = ContrastiveLoss(
            opt=opt,
            margin=opt.margin,
            max_violation=opt.max_violation
        )

        # Loss and Optimizer
        self._setup_optimizer()

        self.Eiters = 0
        self.data_parallel = False

    def _setup_optimizer(self):
        """Setup optimizer with appropriate parameter groups"""
        decay_factor = 1e-4

        if self.opt.precomp_enc_type == 'basic':
            if self.opt.optim == 'adam':
                all_text_params = list(self.txt_enc.parameters())
                bert_params = list(self.txt_enc.bert.parameters())
                bert_params_ptr = [p.data_ptr() for p in bert_params]
                text_params_no_bert = [p for p in all_text_params if p.data_ptr() not in bert_params_ptr]

                param_groups = [
                    {'params': text_params_no_bert, 'lr': self.opt.learning_rate},
                    {'params': bert_params, 'lr': self.opt.learning_rate * 0.1},
                    {'params': self.img_enc.parameters(), 'lr': self.opt.learning_rate},
                ]

                self.optimizer = torch.optim.AdamW(
                    param_groups,
                    lr=self.opt.learning_rate,
                    weight_decay=decay_factor
                )
            elif self.opt.optim == 'sgd':
                all_params = list(self.txt_enc.parameters()) + list(self.img_enc.parameters())
                self.optimizer = torch.optim.SGD(all_params, lr=self.opt.learning_rate, momentum=0.9)
            else:
                raise ValueError(f'Invalid optim option {self.opt.optim}')
        else:
            if self.opt.optim == 'adam':
                all_text_params = list(self.txt_enc.parameters())
                bert_params = list(self.txt_enc.bert.parameters())
                bert_params_ptr = [p.data_ptr() for p in bert_params]
                text_params_no_bert = [p for p in all_text_params if p.data_ptr() not in bert_params_ptr]

                param_groups = [
                    {'params': text_params_no_bert, 'lr': self.opt.learning_rate},
                    {'params': bert_params, 'lr': self.opt.learning_rate * 0.1},
                    {'params': self.img_enc.backbone.top.parameters(),
                     'lr': self.opt.learning_rate * self.opt.backbone_lr_factor},
                    {'params': self.img_enc.backbone.base.parameters(),
                     'lr': self.opt.learning_rate * self.opt.backbone_lr_factor},
                    {'params': self.img_enc.image_encoder.parameters(), 'lr': self.opt.learning_rate},
                ]

                self.optimizer = torch.optim.AdamW(
                    param_groups,
                    lr=self.opt.learning_rate,
                    weight_decay=decay_factor
                )

            elif self.opt.optim == 'sgd':
                param_groups = [
                    {'params': self.txt_enc.parameters(), 'lr': self.opt.learning_rate},
                    {'params': self.img_enc.backbone.parameters(),
                     'lr': self.opt.learning_rate * self.opt.backbone_lr_factor,
                     'weight_decay': decay_factor},
                    {'params': self.img_enc.image_encoder.parameters(), 'lr': self.opt.learning_rate},
                ]

                self.optimizer = torch.optim.SGD(
                    param_groups,
                    lr=self.opt.learning_rate,
                    momentum=0.9,
                    nesterov=True
                )
            else:
                raise ValueError(f'Invalid optim option {self.opt.optim}')

        self.params = list(self.txt_enc.parameters()) + list(self.img_enc.parameters())
        logger.info(f'Use {self.opt.optim} as optimizer, with init lr {self.opt.learning_rate}')

    def forward_emb(self, images, captions, lengths, image_lengths=None):
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            if image_lengths is not None:
                image_lengths = image_lengths.cuda()

        lengths = torch.Tensor(lengths).cuda()

        if self.opt.precomp_enc_type == 'basic':
            img_emb_vse, img_mlp = self.img_enc(images, image_lengths)
        else:
            img_emb_vse, img_mlp = self.img_enc(images)

        cap_emb_vse, cap_lin = self.txt_enc(captions, lengths)

        return img_emb_vse, cap_emb_vse, img_mlp, cap_lin

    def forward_loss(self, img_emb_vse, cap_emb_vse):
        cri_loss = self.criterion(img_emb_vse, cap_emb_vse)
        return cri_loss

    def train_emb(self, images, captions, lengths, image_lengths=None, warmup_alpha=None, tag_index=None, cap_idx=None):
        """One training step given images and captions"""
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # Compute embeddings
        img_emb_vse, cap_emb_vse, img_mlp, cap_lin = self.forward_emb(images, captions, lengths,
                                                                      image_lengths=image_lengths)

        # Compute loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb_vse, cap_emb_vse)
        self.logger.update('loss_vse', loss.item(), img_emb_vse.size(0))

        if tag_index is not None and img_mlp is not None and cap_lin is not None:
            batch_size = cap_lin.size(0)
            max_cap_len = cap_lin.size(1)
            loss_cross_list = []

            for i in range(batch_size):
                noun_indices = tag_index[i]
                if not noun_indices or len(noun_indices) == 0:
                    continue
                noun_indices = [idx for idx in noun_indices if idx < max_cap_len]
                if len(noun_indices) == 0:
                    continue

                cap_noun_vectors = cap_lin[i, noun_indices, :]  # N_nouns x D

                img_positive_vectors = img_mlp[i]  # K x D

                cap_noun_normalized = F.normalize(cap_noun_vectors, p=2, dim=1)  # N_nouns x D
                img_positive_normalized = F.normalize(img_positive_vectors, p=2, dim=1)  # K x D

                similarity = torch.mm(cap_noun_normalized, img_positive_normalized.t())

                max_sim_indices = torch.argmax(similarity, dim=1)  # N_nouns

                most_similar_img_vectors = img_positive_vectors[max_sim_indices]  # N_nouns x D

                diff_vectors = cap_noun_vectors - most_similar_img_vectors  # N_nouns x D

                diff_normalized = l1norm(diff_vectors, dim=1)  # N_nouns x D

                loss_cross_sample = (diff_normalized.abs().sum())**2
                loss_cross_list.append(loss_cross_sample)

            if len(loss_cross_list) > 0:
                loss_cross = torch.stack(loss_cross_list).mean()
                loss = loss + loss_cross
                self.logger.update('loss_cross', loss_cross.item(), img_emb_vse.size(0))

        if warmup_alpha is not None:
            loss = loss * warmup_alpha

        self.logger.update('Le_Total', loss.data.item(), img_emb_vse.size(0))
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

    def train_start(self):
        """Switch to train mode"""
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """Switch to evaluate mode"""
        self.img_enc.eval()
        self.txt_enc.eval()

    def state_dict(self):
        """Return state dict for saving"""
        state_dict = [
            self.img_enc.state_dict(),
            self.txt_enc.state_dict(),
        ]
        return state_dict

    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.txt_enc.load_state_dict(state_dict[1], strict=False)

    def make_data_parallel(self):
        """Enable data parallelism"""
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.data_parallel = True
        logger.info('Both VSE models are data paralleled now.')

    @property
    def is_data_parallel(self):
        return self.data_parallel

    def set_max_violation(self, max_violation):
        """Set max violation mode for both loss functions"""
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()