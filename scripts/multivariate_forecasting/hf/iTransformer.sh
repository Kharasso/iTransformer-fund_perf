export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# concat all funds vertically

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/hf/ \
  --data_path event_driven_mix_ex_10_feature.csv \
  --model_id event_driven_mix_ex_10_feature_48_3 \
  --model $model_name \
  --data hf \
  --features M \
  --seq_len 48 \
  --pred_len 3 \
  --e_layers 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --freq m \
  --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/hf/ \
#   --data_path all_mix_ex_23_feature.csv \
#   --model_id all_mix_ex_23_feature_48_3 \
#   --model $model_name \
#   --data hf \
#   --features MS \
#   --seq_len 48 \
#   --pred_len 3 \
#   --e_layers 3 \
#   --enc_in 23 \
#   --dec_in 23 \
#   --c_out 23 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 256 \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/hf/ \
#   --data_path event_driven_80funds.csv \
#   --model_id event_driven_80funds_512_3 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --batch_size 32 \
#   --seq_len 32 \
#   --pred_len 3 \
#   --e_layers 3 \
#   --enc_in 1440 \
#   --dec_in 1440 \
#   --c_out 1440 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/hf/ \
#   --data_path all_mix.csv \
#   --model_id all_mix_96_3 \
#   --model $model_name \
#   --data hf \
#   --features M \
#   --seq_len 96 \
#   --pred_len 3 \
#   --e_layers 4 \
#   --enc_in 18 \
#   --dec_in 18 \
#   --c_out 18 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 256 \
#   --itr 1

# concat all funds horizontally
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/hf/ \
#   --data_path event_driven_80funds.csv \
#   --model_id event_driven_48_2 \
#   --model $model_name \
#   --data hf \
#   --features M \
#   --seq_len 48 \
#   --pred_len 3 \
#   --e_layers 4 \
#   --enc_in 18 \
#   --dec_in 18 \
#   --c_out 18 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --learning_rate 0.001 \
#   --itr 1

