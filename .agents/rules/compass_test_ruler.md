# How to run test_ruler.py with COMPASS hyperparameters
When we need to tune COMPASS hyperparameters (top-p and lambda), we DO NOT hardcode them in `nanovllm/config.py` anymore.
Instead, use the explicitly added CLI arguments for `test_ruler.py`: `--compass-top-p` and `--compass-lambda`.

Example of how to sweep parameters and run:
```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/zijie/Code/nano-vllm:$PYTHONPATH python tests/test_ruler.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --data-dir tests/data/ruler_32k \
    --datasets niah_single_1 \
    --num-samples 1 \
    --max-model-len 32768 \
    --enable-offload \
    --sparse-policy COMPASS \
    --compass-top-p 0.5 \
    --compass-lambda 0.01
```

- `--compass-top-p`: Determines the threshold to select CPU sub-blocks (tuning GPU KV cache IO density). Lower values mean sparser IO, possibly hurting accuracy.
- `--compass-lambda`: Determines the absolute threshold for selecting global chunks from the initial full CPU phase (the higher the lambda, the smaller the initial pool of candidate sub-blocks).

Do not change `conifg.py` values to achieve this unless globally intended out of test_ruler scope.
