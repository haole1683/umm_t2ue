# Folder Name Rule: DatasetName_Method_ClipVersion_{Pretrain/Scratch}
# Noise Train Folder DatasetName_Method_ClipVersion_{Pretrain/Scratch}_Train

###################################################

# Flickr8k
## 1.Clean Training on CLIP RN50 Scratch     DOING
python -m src.main --name flickr8k_clean_rn50_scratch --train_data ./data/Flicker-8k/train.csv --eval_test_data_dir ./data/Flicker-8k/test.csv --image_key images  --caption_key caption  --device_id 0 --model_name RN50
## 2.Random Noise Training on CLIP RN50 Scratch     DOING
python -m src.main_my_poison --name flickr8k_Random_rn50_scratch --train_data ./data/Flicker-8k/train.csv --eval_test_data_dir ./data/Flicker-8k/test.csv --image_key images  --caption_key caption  --device_id 0 --lr 1e-5 --model_name RN50 --noise_path /remote-home/songtianwei/research/unlearn_mm/Multimodal-Unlearnable-Examples/flickr8k_random_noise_norm.pt
## 2.EM Poison Training on CLIP RN50 Scratch
### Step1     DOING
python -m src.poison2 --name flickr8k_EM_rn50_scratch_train --train_data ./data/Flicker-8k/train.csv --image_key images  --caption_key caption  --device_id 0  --lr 1e-4 --model_name RN50
### Step2
python -m src.poison_main --name flickr8k_EM_rn50_scratch --train_data ./data/Flicker-8k/train.csv --eval_test_data_dir ./data/Flicker-8k/test.csv --image_key images  --caption_key caption  --device_id 0 --save_pert flickr8k_EM_rn50_scratch_train  --lr 5e-4 --model_name RN50
## 3.MEM3 Poison Training on CLIP RN50 Scratch
### Step1     DOING
python -m src.poison --name flickr8k_MEM3_rn50_scratch_train --train_data ./data/Flicker-8k/train.csv  --image_key images  --caption_key caption  --device_id 0 --token_num 3 --lr 1e-4 --model_name RN50
### Step2
python -m src.poison_main --name flickr8k_MEM3_rn50_scratch --train_data ./data/Flicker-8k/train.csv --eval_test_data_dir ./data/Flicker-8k/test.csv --image_key images  --caption_key caption  --device_id 0 --save_pert flickr8k_MEM3_rn50_scratch_train --token_num 3  --lr 5e-4 --model_name RN50
## 4.MEM5 Poison Training on CLIP RN50 Scratch
### Step1
python -m src.poison --name flickr8k_MEM3_rn50_scratch_train --train_data ./data/Flicker-8k/train.csv  --image_key images  --caption_key caption  --device_id 0 --token_num 3 --lr 1e-4 --model_name RN50
### Step2
python -m src.poison_main --name flickr8k_MEM3_rn50_scratch --train_data ./data/Flicker-8k/train.csv --eval_test_data_dir ./data/Flicker-8k/test.csv --image_key images  --caption_key caption  --device_id 0 --save_pert flickr8k_MEM3_rn50_scratch_train --token_num 3  --lr 5e-4 --model_name RN50
## 5.T2UE(Ours) Poison Training on CLIP RN50 Scratch
### Step1     Done
TODO
### Step2     DOING
python -m src.main_my_poison --name test2_flickr8k_T2UE_rn50_scratch --train_data ./data/Flicker-8k/train.csv --eval_test_data_dir ./data/Flicker-8k/test.csv --image_key images  --caption_key caption  --device_id 0 --lr 1e-5 --model_name RN50 --noise_path /remote-home/songtianwei/research/unlearn_multimodal/outputCLIP/unlearn_stage2_generate_noise_temp1/ViT-B-32/Flickr-8K/classWise/noise_gen2_30000-224-224_flickr8K_ViT-B-32.pt


#####################################################


