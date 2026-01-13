# Commands

## Running COMPASS (with PYTHONPATH)

For development, use PYTHONPATH instead of pip install:

```bash
# Set PYTHONPATH
export PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH

# Activate conda environment
conda activate ruler

# Run RULER benchmark
./scripts/run_ruler.sh llama3.1-8b-chat synthetic full

# Or run directly
cd eval/RULER/scripts
bash run.sh llama3.1-8b-chat synthetic --metric full
```

## Available Metrics

| Metric | Description |
|--------|-------------|
| `full` | Full attention (baseline) |
| `xattn` | X-attention sparse |
| `avgpool` | Average pooling sparse |
| `minfer` | Minference |
| `compass` | COMPASS method |
| `flex` | FlexPrefill |

## Run Specific Tasks

```bash
# Single task
./scripts/run_ruler.sh llama3.1-8b-chat synthetic full --task niah_single_1

# Multiple tasks
./scripts/run_ruler.sh llama3.1-8b-chat synthetic full --task niah_single_1,vt,qa_1
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONDA_ENV` | `ruler` | Conda environment name |
| `MODEL_DIR` | `/home/zijie/models` | Model weights directory |
| `CUDA_VISIBLE_DEVICES` | - | GPU selection |

## Import Test

```bash
# Test compass imports
PYTHONPATH=/home/zijie/Code/COMPASS:$PYTHONPATH python -c "from compass.src.Compass import Compass; print('OK')"
```

## Dataset Download

```bash
cd eval/RULER
bash setup.sh
```

Datasets (auto-downloaded):
- `PaulGrahamEssays.json` (~3MB)
- `hotpotqa.json` (~46MB)
- `squad.json` (~4MB)
