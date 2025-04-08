CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_ea_answer_llama2chat \
    --ea-model-path HArmonizedSS/HASS-LLaMA2-Chat-7B \
    --base-model-path meta-llama/Llama-2-7b-chat-hf \
    --model-id llama2-chat-7b_hass \
    --bench-name mt_bench \
    --temperature 0 \
    --seed 42
