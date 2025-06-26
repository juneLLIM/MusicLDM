CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./src python style_transfer.py \
    --proj-name noiseAdaIN-style-transfer-text \
    --original "japanese animation bgm orchestra" \
    "japanese animation bgm orchestra" \
    --reference "rock music with hard distortion" \
    --seed 78949 \
    --config-path musicldm.yaml \
    --cache-dir cache \
    --out-dir log \
    --batch-size 1 \
    --log-dir log
