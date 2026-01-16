# flickr
python find_hardnegative.py --model_path runs/f30k_vse1_butd_region_bert/model_best.pth --data_name f30k --data_path './data/f30k' --batch_size 128 --output_dir hard_negative_flickr --workers 0 --model VSE_S1

# coco
python find_hardnegative.py --model_path runs/coco_2nd_butd_region_bert/model_best.pth --data_name coco --data_path './data/coco' --batch_size 128 --output_dir hard_negative_coco --workers 0 --model VSE_S1
