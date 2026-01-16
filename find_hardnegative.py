"""Find and record hard negatives from train_loader"""
import os
import json
import numpy as np
import torch
from transformers import BertTokenizer

from lib.datasets import image_caption
from lib.vse import VSEModel_S2, VSEModel_S1

import logging
import arguments


def find_hard_negatives(opt, train_loader, model):
    logger = logging.getLogger(__name__)
    model.val_start()

    hard_negatives_dict = {}

    with torch.no_grad():
        for i, train_data in enumerate(train_loader):
            if opt.precomp_enc_type == 'basic':
                images, img_lengths, captions, lengths, cap_idx, tag_index, img_ids = train_data
            else:
                images, captions, lengths, cap_idx, tag_index, img_ids = train_data

            if opt.model == 'VSE':
                if opt.precomp_enc_type == 'basic':
                    img_emb, cap_emb, _, _ = model.forward_emb(images, captions, lengths, image_lengths=img_lengths)
                else:
                    img_emb, cap_emb, _, _ = model.forward_emb(images, captions, lengths)
            elif opt.model == 'DualVSE':
                if opt.precomp_enc_type == 'basic':
                    img_emb, cap_emb, _, _ = model.forward_emb(images, captions, lengths, image_lengths=img_lengths)
                else:
                    img_emb, cap_emb, _, _ = model.forward_emb(images, captions, lengths)

            img_emb_norm = img_emb / img_emb.norm(dim=1, keepdim=True)
            cap_emb_norm = cap_emb / cap_emb.norm(dim=1, keepdim=True)
            similarity_matrix = torch.matmul(img_emb_norm, cap_emb_norm.t()).cpu().numpy()
            batch_size = similarity_matrix.shape[0]
            if torch.is_tensor(img_ids):
                img_ids_list = img_ids.cpu().numpy().tolist()
            else:
                img_ids_list = list(img_ids)

            for idx in range(batch_size):
                current_img_id = img_ids_list[idx]
                positive_sim = similarity_matrix[idx, idx]
                i2t_sims = similarity_matrix[idx].copy()
                max_sim_in_batch = np.max(i2t_sims)

                if positive_sim < max_sim_in_batch:
                    i2t_sims[idx] = -float('inf')
                    hard_neg_ids = np.argmax(i2t_sims)
                    hard_neg_img_id = img_ids_list[int(hard_neg_ids)]

                    if current_img_id != hard_neg_img_id:
                        if current_img_id not in hard_negatives_dict:
                            hard_negatives_dict[current_img_id] = []
                        hard_negatives_dict[current_img_id].append(hard_neg_img_id)

            if (i + 1) % opt.log_step == 0:
                logger.info(f'Processed {i + 1}/{len(train_loader)} batches, found {len(hard_negatives_dict)} hard negative samples')

    return hard_negatives_dict


def main():
    # Hyper Parameters
    parser = arguments.get_argument_parser()
    parser.add_argument('--model_path', default='', type=str,
                        help='Path to trained model')
    parser.add_argument('--output_dir', default='hard_negatives', type=str,
                        help='Directory to save hard negatives')
    opt = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(opt)

    # Load Tokenizer and Vocabulary
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)

    # Load data
    train_loader, val_loader = image_caption.get_loaders(
        opt.data_path, opt.data_name, tokenizer, opt.batch_size, opt.workers, opt)

    # Construct the model
    if opt.model == 'VSE_S2':
        model = VSEModel_S2(opt)
    elif opt.model == 'VSE_S1':
        model = VSEModel_S1(opt)

    # Load trained model
    if opt.model_path and os.path.isfile(opt.model_path):
        logger.info(f"Loading checkpoint from '{opt.model_path}'")
        checkpoint = torch.load(opt.model_path, weights_only=False)

        if not model.is_data_parallel:
            model.make_data_parallel()

        model.load_state_dict(checkpoint['model'])
        logger.info(f"Loaded checkpoint (epoch {checkpoint['epoch']}, best_rsum {checkpoint['best_rsum']})")
    else:
        logger.error(f"No checkpoint found at '{opt.model_path}'")
        return

    if not model.is_data_parallel:
        model.make_data_parallel()

    model.val_start()

    # Find hard negatives
    logger.info("Starting to find hard negatives...")
    hard_negatives_dict = find_hard_negatives(opt, train_loader, model)

    # Save results
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    output_path = os.path.join(opt.output_dir, 'hard_negatives_train.json')
    with open(output_path, 'w') as f:
        json.dump(hard_negatives_dict, f, indent=2)

    logger.info(f"Saved {len(hard_negatives_dict)} hard negative records to {output_path}")
    logger.info("\n" + "="*50)
    logger.info("Statistics:")
    logger.info(f"Total hard negative samples found: {len(hard_negatives_dict)}")
    logger.info(f"Total samples in train set: {len(train_loader.dataset)}")
    logger.info(f"Percentage of samples with hard negatives: {100 * len(hard_negatives_dict) / len(train_loader.dataset):.2f}%")
    logger.info("\nExample records (first 5):")
    for i, (pos_cap_idx, neg_cap_idx) in enumerate(list(hard_negatives_dict.items())[:5]):
        logger.info(f"  Positive cap_idx: {pos_cap_idx} -> Hard negative cap_idx: {neg_cap_idx}")
    logger.info("="*50)


if __name__ == '__main__':
    main()