# just for collection, you can run each of them directly and individually in the cmd

# training scripts
python -u train.py --config reproduce_config/train_denet_coco_fold0.yml
python -u train.py --config reproduce_config/train_denet_coco_fold1.yml
python -u train.py --config reproduce_config/train_denet_coco_fold2.yml
python -u train.py --config reproduce_config/train_denet_coco_fold3.yml

# testing scripts
python -u test.py --config reproduce_config/test_denet_coco_fold0_1way_1shot.yml
python -u test.py --config reproduce_config/test_denet_coco_fold0_1way_5shot.yml
python -u test.py --config reproduce_config/test_denet_coco_fold1_1way_1shot.yml
python -u test.py --config reproduce_config/test_denet_coco_fold1_1way_5shot.yml
python -u test.py --config reproduce_config/test_denet_coco_fold2_1way_1shot.yml
python -u test.py --config reproduce_config/test_denet_coco_fold2_1way_5shot.yml
python -u test.py --config reproduce_config/test_denet_coco_fold3_1way_1shot.yml
python -u test.py --config reproduce_config/test_denet_coco_fold3_1way_5shot.yml