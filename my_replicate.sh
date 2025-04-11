CUDA_VISIBLE_DEVICES=2,3 python -m evaluation.gen_ea_answer_llama2chat \
    --ea-model-path /home/tk/Desktop/HASS/ckpt/state_3 \
    --base-model-path meta-llama/Llama-2-7b-chat-hf \
    --model-id llama2-chat-7b_hass \
    --bench-name mt_bench \
    --temperature 0 \
    --seed 42

#   --ea-model-path HArmonizedSS/HASS-LLaMA2-Chat-7B \

#CUDA_VISIBLE_DEVICES=4 accelerate launch -m --mixed_precision=bf16 train.main_hass \
#    --tmpdir /mnt/wd_ssd/sharegpt_0_67999_mufp16/ \
#    --cpdir ./ckpt2 \
#    --configpath ./train/llama_2_chat_7B_config.json \
#    --basepath ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590 \
#    --epoch 10 \
#    --bs 1 --gradient-accumulation-steps 2 \
#    --topk 10 \
#    --topk_w 0 \
#    --forward_num_total 3
