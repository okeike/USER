"""COCO dataset loader"""
import torch
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
import random
import json
import logging

logger = logging.getLogger(__name__)

class PrecompRegionDataset(data.Dataset):
    """
    Load precomputed captions and image features for COCO or Flickr
    """

    def __init__(self, data_path, data_name, data_split, tokenizer, opt, train):
        self.tokenizer = tokenizer
        self.opt = opt
        self.train = train
        self.data_path = data_path
        self.data_name = data_name

        loc_cap = osp.join(data_path, 'precomp')
        loc_image = osp.join(data_path, 'precomp')

        # Captions
        self.captions = []
        with open(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())

        #  Caption Tags
        self.tag_indexes = []
        with open(osp.join(loc_cap, '%s_tags.txt' % data_split), 'r') as f:
            for line in f:
                n_tag_index = []
                pos_tag = line.strip().split(' ')
                for i in range(len(pos_tag)):
                    if pos_tag[i] == 'NN' or pos_tag[i] == 'NNS' or pos_tag[i] == 'NNP':
                        n_tag_index.append(i)
                self.tag_indexes.append(n_tag_index)

        # Captions Ids
        self.ids = []
        with open(osp.join(loc_cap, '%s_ids.txt' % data_split), 'r') as f:
            for line in f:
                self.ids.append(line.strip())

        # Image features
        self.images = np.load(os.path.join(loc_image, '%s_ims.npy' % data_split))

        # Image Triplets features
        self.image_sgg = []
        with open(osp.join(loc_cap, '%s_sgg.txt' % data_split), 'r') as f:
            for line in f:
                self.image_sgg.append(line.strip())
        # Triplets vocab
        if opt.data_name == 'f30k':
            vocab_path = 'data/f30k/triplet_vocab.json'
        elif opt.data_name == 'coco':
            vocab_path = 'data/coco/triplet_vocab.json'
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            self.word2idx = vocab_data['word2idx']

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        num_images = len(self.images)

        if num_images != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

        print(data_split, " ", "image length", num_images, " ", "caption length:", self.length, " ", "length of sgg:", len(self.image_sgg))
    def __getitem__(self, index):
        # handle the image redundancy
        img_index = index // self.im_div
        caption = self.captions[index]
        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)
        tag_index = self.tag_indexes[index]
        img_id = self.ids[img_index]

        # Convert caption (string) to word ids (with Size Augmentation at training time)
        target = process_caption(self.tokenizer, caption_tokens, self.train)
        image = self.images[img_index]
        if self.train:  # Size augmentation for region feature
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            image = image[np.where(rand_list > 0.20)]
        # image features
        image = torch.Tensor(image)
        # image triplets
        triplets_string = self.image_sgg[img_index]
        triplets_indices = triplets_process(triplets_string, self.word2idx)

        return image, target, index, img_index, tag_index, img_id, triplets_indices

    def __len__(self):
        return self.length


def process_caption(tokenizer, tokens, train=True):
    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()

        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% randomly change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
            # 10% randomly change token to random token
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    deleted_idx.append(len(output_tokens) - 1)
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
    target = tokenizer.convert_tokens_to_ids(output_tokens)
    target = torch.Tensor(target)
    return target

def triplets_process(triplets_string, word2idx):
    """
    turn triplets into a 2d list
    Args:
        triplets_string: "dog chasing cat, man riding bike, woman holding umbrella"
        word2idx

    Returns:
        triplet_indices
    """
    sentences = [s.strip() for s in triplets_string.split(',') if s.strip()]
    triplet_indices = []
    unk_idx = word2idx.get('[UNK]', 1)

    for sentence in sentences:
        words = sentence.split()
        indices = [word2idx.get(word.lower(), unk_idx) for word in words]
        triplet_indices.append(indices)

    return triplet_indices


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, captions, ids, img_ids, tag_index, img_ids, img_sggs = zip(*data)
    if len(images[0].shape) == 2:  # region feature
        # Sort a data list by caption length
        # Merge images (convert tuple of 3D tensor to 4D tensor)
        # images = torch.stack(images, 0)
        img_lengths = [len(image) for image in images]
        all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
        for i, image in enumerate(images):
            end = img_lengths[i]
            all_images[i, :end] = image[:end]
        img_lengths = torch.Tensor(img_lengths)

        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()

        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        # Merge triplets
        num_tri = []
        tri_lengths_all = []

        for sample in img_sggs:
            num_tri.append(len(sample))
            sample_lengths = [len(triplet) for triplet in sample]
            tri_lengths_all.extend(sample_lengths)

        max_num_tri = max(num_tri)
        max_tri_len = max(tri_lengths_all)

        img_tri = torch.zeros(len(img_sggs), max_num_tri, max_tri_len).long()  # 或用.int()

        tri_length = []

        for i, sample in enumerate(img_sggs):
            sample_tri_lengths = []

            for j, triplet in enumerate(sample):
                triplet_len = len(triplet)
                sample_tri_lengths.append(triplet_len)

                for k, idx in enumerate(triplet):
                    img_tri[i, j, k] = idx

            tri_length.append(sample_tri_lengths)

        return all_images, img_lengths, targets, lengths, ids, tag_index, img_ids, img_tri, tri_length


def get_loader(data_path, data_name, data_split, tokenizer, opt, batch_size=100,
               shuffle=True, num_workers=2, train=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if train:
        drop_last = True
    else:
        drop_last = False

    dset = PrecompRegionDataset(data_path, data_name, data_split, tokenizer, opt, train)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers,
                                              drop_last=drop_last)

    return data_loader


def get_loaders(data_path, data_name, tokenizer, batch_size, workers, opt):
    train_loader = get_loader(data_path, data_name, 'train', tokenizer, opt,
                              batch_size, True, workers)
    val_loader = get_loader(data_path, data_name, 'dev', tokenizer, opt,
                            batch_size, False, workers, train=False)
    return train_loader, val_loader


def get_train_loader(data_path, data_name, tokenizer, batch_size, workers, opt, shuffle):
    train_loader = get_loader(data_path, data_name, 'train', tokenizer, opt,
                              batch_size, shuffle, workers)
    return train_loader


def get_test_loader(split_name, data_name, tokenizer, batch_size, workers, opt):
    test_loader = get_loader(opt.data_path, data_name, split_name, tokenizer, opt,
                             batch_size, False, workers, train=False)
    return test_loader
