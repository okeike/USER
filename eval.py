import os
import argparse
import logging
from lib import evaluation

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coco',
                        help='coco or f30k')
    parser.add_argument('--data_path', default='/tmp/data/coco')
    parser.add_argument('--save_results', type=bool, default=False)
    parser.add_argument('--model', type=str, default='VSE_S1', help='VSE_S1 or VSE_S2')
    opt = parser.parse_args()

    if opt.dataset == 'coco':
        weights_bases = [
            'runs/coco_1st_butd_region_bert',
            'runs/coco_2nd_butd_region_bert'
        ]
    elif opt.dataset == 'f30k':
        weights_bases = [
            'runs/f30k_1st_butd_region_bert',
            'runs/f30k_2nd_butd_region_bert'
        ]
    else:
        raise ValueError('Invalid dataset argument {}'.format(opt.dataset))

    for base in weights_bases:
        logger.info('Evaluating {}...'.format(base))
        model_path = os.path.join(base, 'model_best.pth')
        if opt.save_results:  # Save the final results for computing ensemble results
            save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset))
        else:
            save_path = None

        if opt.dataset == 'coco':
            if opt.model == 'VSE_S2':
                evaluation.evalrank_s2(model_path, data_path=opt.data_path, split='testall', fold5=True, save_path=save_path)
                evaluation.evalrank_s2(model_path, data_path=opt.data_path, split='testall', fold5=False, save_path=save_path)
            elif opt.model == 'VSE_S1':
                evaluation.evalrank_s1(model_path, data_path=opt.data_path, split='testall', fold5=True, save_path=save_path)
                evaluation.evalrank_s1(model_path, data_path=opt.data_path, split='testall', fold5=False, save_path=save_path)

        elif opt.dataset == 'f30k':
            if opt.model == 'VSE_S2':
                evaluation.evalrank_s2(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)
            elif opt.model == 'VSE_S1':
                evaluation.evalrank_s1(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)

if __name__ == '__main__':
    main()
