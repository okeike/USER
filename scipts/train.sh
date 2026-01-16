'''Flickr'''
# vse1
python train.py --data_name 'f30k' --data_path './data/f30k' --model 'VSE_S1' --logger_name runs/f30k_vse1_butd_region_bert/log --model_name runs/f30k_vse1_butd_region_bert --num_epochs=25 --lr_update=15 --learning_rate=.0005--workers 0 --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1
# vse2
python train.py --data_name 'f30k' --data_path './data/f30k' --model 'VSE_S2' --logger_name runs/f30k_vse2_butd_region_bert/log --model_name runs/f30k_vse2_butd_region_bert --num_epochs=25 --lr_update=15 --learning_rate=.0005 --workers 0 --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1

'''COCO'''
# vse1
python train.py --data_name 'coco' --data_path './data/coco' --model 'VSE_S1' --logger_name runs/coco_1st_butd_region_bert/log --model_name runs/coco_1st_butd_region_bert --num_epochs=25 --lr_update=15 --learning_rate=.0005 --workers 0 --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1
# vse2
python train.py --data_name 'coco' --data_path './data/coco' --model 'VSE_S2' --logger_name runs/coco_2nd_butd_region_bert/log --model_name runs/coco_2nd_butd_region_bert --num_epochs=25 --lr_update=15 --learning_rate=.0005 --workers 0 --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1
