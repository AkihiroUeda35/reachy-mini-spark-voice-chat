cd ./spark-vllm-docker
./launch-cluster.sh --solo \
exec vllm serve sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP \
    --trust-remote-code \
    --served-model-name spark \
    --quantization modelopt \
    --max-model-len 262144 \
    --max-num-seqs 4 \
    --port 8010 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.6 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":4,"moe_backend":"triton"}'


