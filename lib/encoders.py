"""VSE modules"""
import json

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from transformers import BertModel
from transformers.models import opt

from lib.modules.resnet import ResnetFeatureExtractor
from lib.modules.aggr.gpo import GPO
from lib.modules.mlp import MLP

import logging

logger = logging.getLogger(__name__)


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


def get_text_encoder(embed_size, no_txtnorm=False):
    return EncoderText(embed_size, no_txtnorm=no_txtnorm)


def get_image_encoder(img_dim, embed_size, no_imgnorm=False, opt=None):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """

    if opt.model == 'VSE':
        img_enc = EncoderImageAggr_new(img_dim, embed_size, no_imgnorm, opt)
    else:
        img_enc = EncoderImageAggr(img_dim, embed_size, no_imgnorm)

    return img_enc


class EncoderImageAggr_new(nn.Module):
    def __init__(self, img_dim, embed_size, no_imgnorm=False, opt=None):
        super(EncoderImageAggr_new, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)

        if opt.data_name == 'f30k':
            vocab_path = "data/f30k/triplet_vocab.json"
            checkpoint = "data/f30k/embedding_init_weights.pth"
        elif opt.data_name == 'coco':
            vocab_path = "data/coco/triplet_vocab.json"
            checkpoint = "data/coco/embedding_init_weights.pth"
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)['word2idx']
        pretrained_weights = torch.load(checkpoint)["weight"]

        vocab_size = len(vocab)
        self.triplet_embedding = nn.Embedding(vocab_size, 300, padding_idx=0)
        self.triplet_embedding.weight.data.copy_(pretrained_weights)

        self.triplet_mlp = MLP(300, embed_size // 2, embed_size, 2)

        self.gpool = GPO(32, 32)
        self.gpool_tri = GPO(32, 32)

        self.output_mlp = MLP(embed_size + 300, embed_size // 2, embed_size, 1)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths, img_tri, tri_length):
        """
        Extract image feature vectors with triplet information.

        Args:
            images: (batch_size, num_regions, img_dim) 原始图像区域特征
            image_lengths: (batch_size,) 每个样本的有效区域数
            img_tri: (batch_size, max_num_tri, max_tri_len) 三元组词索引，填充值为0
            tri_length: list of list，tri_length[i][j]是第i个样本第j个三元组的长度

        Returns:
            features: (batch_size, embed_size) 最终的图像表示
            mlp_features: 用于后续处理的中间特征
        """
        batch_size = images.size(0)
        features = self.fc(images)

        features = self.mlp(images) + features
        gpool_img_features, img_pool_weights = self.gpool(features, image_lengths)

        # Triplets
        triplet_emb = self.triplet_embedding(img_tri)
        mask = (img_tri == 0)
        mask = mask.unsqueeze(-1)
        triplet_emb_masked = triplet_emb.clone()
        triplet_emb_masked[mask.expand_as(triplet_emb)] = -1e9
        triplet_features, _ = triplet_emb_masked.max(dim=2)
        tri_lengths_tensor = torch.ones(batch_size, dtype=torch.long).to(triplet_features.device)
        for i in range(batch_size):
            if len(tri_length[i]) != 0:
                tri_lengths_tensor[i] = len(tri_length[i])
            else:
                continue

        gpool_tri_features, tri_pool_weights = self.gpool_tri(triplet_features, tri_lengths_tensor)

        # Combine
        combined_gpool = torch.cat([gpool_img_features, gpool_tri_features], dim=-1)
        combined_gpool_expanded = combined_gpool.unsqueeze(1)  # (batch_size, 1, 1324)
        gpool_features = self.output_mlp(combined_gpool_expanded)  # (batch_size, 1, 1024)
        gpool_features = gpool_features.squeeze(1)  # (batch_size, 1024)

        # ==================== L2 normalization ====================
        if not self.no_imgnorm:
            gpool_features = l2norm(gpool_features, dim=-1)

        return gpool_features, features

class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
        self.gpool = GPO(32, 32)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths):
        """Extract image feature vectors."""
        features = self.fc(images)
        # When using pre-extracted region features, add an extra MLP for the embedding transformation
        features = self.mlp(images) + features
        mlp_features = features

        features, pool_weights = self.gpool(features, image_lengths)

        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features, mlp_features

# Language Model with BERT
class EncoderText(nn.Module):
    def __init__(self, embed_size, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, embed_size)
        self.gpool = GPO(32, 32)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        bert_emb = self.bert(x, bert_attention_mask)[0]  # B x N x D
        cap_len = lengths

        cap_emb = self.linear(bert_emb)
        cap_lin = cap_emb

        pooled_features, pool_weights = self.gpool(cap_emb, cap_len.to(cap_emb.device))

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            pooled_features = l2norm(pooled_features, dim=-1)

        return pooled_features, cap_lin