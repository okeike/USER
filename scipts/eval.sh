'''Flickr'''
# vse1
python eval.py --dataset 'f30k' --data_path './data/f30k' --model 'VSE_S1' --save_results True
# vse2
python eval.py --dataset 'f30k' --data_path './data/f30k' --model 'VSE_S2' --save_results True

'''COCO'''
# vse1
python eval.py --dataset 'coco' --data_path './data/coco' --model 'VSE_S1' --save_results True
# vse2
python eval.py --dataset 'coco' --data_path './data/coco' --model 'VSE_S2' --save_results True
