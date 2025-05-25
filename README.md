# Large Scale AI Engineering: Flash Attention and FP8 Training Experiments

This project investigates the performance improvements of two key optimizations in transformer training:
1. **Flash Attention**: An efficient implementation of attention that reduces memory usage and improves computational speed by avoiding redundant memory reads/writes.
2. **FP8 Training**: Using 8-bit floating point precision for training, which can significantly reduce memory usage and potentially increase training speed while maintaining model quality.

The project is a part of the [Large Scale AI Engineering course at ETH Zurich](https://ai.ethz.ch/education/lectures-and-seminars/large-scale-ai-engineering.html). These experiments were conducted on the [CSCS Alps cluster](https://www.cscs.ch/computers/alps), which has the Grace Hopper Processors. The 2 selected features are optimized for the Grace Hopper Processors, and thus are valuable to study.

## Project Structure

```
.
├── jobs/                    # Experiment job scripts
│   ├── baseline/           # Baseline model training jobs
│   ├── flash-attn/         # Flash Attention experiment jobs
│   └── fp8/               # FP8 training experiment jobs
├── output/                 # Experiment outputs
│   ├── logs/              # Training logs
│   └── plots/             # Generated plots
├── plot.py                # Script for generating comparison plots
└── README.md
```

## Setting up Environment
To set up the environment, start an interactive job:
```
srun --account=<account_name> --container-writable -p <partition_name> --pty bash
```
Then, build the image specified in `envs/Dockerfile`:
```
podman build -t <image_name> -f envs/Dockerfile
```
Finally, we can save this image in a `.sqsh` file to avoid needing to rebuild the image everytime:
```
enroot import -o envs/base_img.sqsh podman://<image_name>
```
This will store the needed image file in the `envs` folder. All the job scripts provided in this repo point here directly.

## Running Experiments

The project uses SLURM for job scheduling. Each experiment type has its own directory in the `jobs/` folder with submission scripts.

### Running a Baseline Experiment

```bash
sbatch jobs/baseline/submit-baseline.sh
```

### Running a Flash Attention Experiment

```bash
sbatch jobs/flash-attn/submit-flash-attn.sh
```

### Running an FP8 Experiment

```bash
sbatch jobs/fp8/submit-fp8.sh
```

Each experiment will:
1. Train the model with the specified configuration
2. Log metrics to JSONL files in `output/logs/`

## Analyzing Results

After running experiments, use the plotting script to generate comparison plots:

```bash
python plot.py --config plot_config.yaml
```

This will create visualizations comparing:
- Training speed (tokens/second)
- Memory usage
- Model FLOPs utilization (MFU)
- Training time
- Loss curves

Plot configs for flash attention experiments and for fp8 training are provided in the `plot_configs` folder.
