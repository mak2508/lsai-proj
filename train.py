import os
import time
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torchao.float8 import convert_to_float8_training, Float8LinearConfig
from torch.nn.attention import SDPBackend, sdpa_kernel

from dataset import CollatorForCLM, ParquetDataset
from model import Transformer, TransformerModelArgs
from utils import build_lr_scheduler, clip_grad_norm_, get_args, get_num_params, get_num_flop_per_token, init_logger, logger, PRECISION_STR_TO_DTYPE, set_default_dtype


# optional: filter modules from being eligible for float8 conversion
def module_filter_fn(mod: torch.nn.Module, fqn: str):
    # don't convert the last module
    if fqn == "1":
        return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True


def train(args):
  logger.info(f"Experiment args: {args}")
  # Init
  device = torch.device(f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
  model_dtype = PRECISION_STR_TO_DTYPE[args.model_dtype]

  # Set up metrics logging
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  os.makedirs(args.output_dir, exist_ok=True)
  metrics_file = os.path.join(args.output_dir, f"{args.job_name}_{timestamp}.jsonl")
  metrics_data = []
  
  # Set up DataLoader
  logger.info("Setting up DataLoaders...")
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
  train_ds = ParquetDataset(args.dataset, tokenizer, args.sequence_length, args.batch_size*args.training_steps)
  train_collator = CollatorForCLM(args.sequence_length, tokenizer.pad_token_id)
  train_dl = DataLoader(train_ds,
                        batch_size=args.batch_size,
                        collate_fn=train_collator)
  train_dl_iterator = iter(train_dl)

  # Set up Model
  logger.info("Setting up Model...")
  model_config = TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        vocab_size=tokenizer.vocab_size,
        seq_len=args.sequence_length,
        flash_attn3=args.attention_backend == "flash-attn3",
    )
  with set_default_dtype(model_dtype):
    model = Transformer(model_config).to(device)
  
  if args.fp8_train:
    logger.info(f"Converting model to float8, using recipe: {args.fp8_recipe}")
    config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
    convert_to_float8_training(model, module_filter_fn=module_filter_fn, config=config)
  
  if args.compile:
    logger.info("Using `torch.compile`")
    model = torch.compile(model, fullgraph=True)
  
  model.train()

  # Build Optimizers & LR Scheduler
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=args.fused_optimizer)
  lr_scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps)

  # Utils
  num_flop_per_token = get_num_flop_per_token(
      get_num_params(model, exclude_embedding=True),
      model_config,
  )

  if args.attention_backend == "cudnn":
    backend = SDPBackend.CUDNN_ATTENTION
  elif args.attention_backend == "efficient":
    backend = SDPBackend.EFFICIENT_ATTENTION
  elif args.attention_backend == "flash-attn":
    backend = SDPBackend.FLASH_ATTENTION
  else:
    backend = None

  def training_loop():
    ntokens_since_last_log = 0
    ntraining_tokens_since_last_log = 0
    time_last_log = time.perf_counter()

    logger.info("Starting training!")
    train_step = 0
    while train_step < args.training_steps:
      train_step += 1

      # Profiling
      if args.profile and args.profile_step_start == train_step:
        torch.cuda.cudart().cudaProfilerStart()
        torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

      input_ids, labels = next(train_dl_iterator)
      ntokens_since_last_log += args.batch_size * args.sequence_length
      num_items_in_batch = labels.ne(-100).sum()
      ntraining_tokens_since_last_log += num_items_in_batch
      input_ids = input_ids.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()

      logits = model(input_ids)
      loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="sum")
      loss = loss / num_items_in_batch
      del logits
      loss.backward()

      # Clip gradients
      clip_grad_norm_(model.parameters(), args.grad_max_norm)

      optimizer.step()
      lr_scheduler.step()

      # Logging
      if (train_step == 1 or train_step % args.logging_frequency == 0):
        time_delta = time.perf_counter() - time_last_log
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # tokens per second per device, abbreviated as tps
        tps = float(ntokens_since_last_log / time_delta)
        mfu = float(100 * num_flop_per_token * tps / 989e12)
        tflops = float(num_flop_per_token * tps / 1e12)
        training_tps = float(ntraining_tokens_since_last_log / time_delta)
        current_lr = float(lr_scheduler.get_last_lr()[0])

        # Log to console
        logger.info(f"Time: {current_time} | Step: {train_step} | Loss: {loss.item():.2f} | Tokens per second: {tps:.2f} | Training tokens per second (%): {100*training_tps/tps:.2f} | MFU (%): {mfu:.2f} | TFLOPs: {tflops:.2f}")
        
        # Collect metrics in memory - ensure all values are native Python types
        metrics_data.append({
          'timestamp': current_time,
          'step': int(train_step),
          'loss': float(loss.item()),
          'tokens_per_second': float(tps),
          'training_tokens_per_second': float(training_tps),
          'training_tokens_percentage': float(100 * training_tps / tps),
          'mfu': float(mfu),
          'tflops': float(tflops),
          'learning_rate': float(current_lr),
          'time_delta': float(time_delta)  # Time since last log
        })

        ntokens_since_last_log = 0
        ntraining_tokens_since_last_log = 0
        time_last_log = time.perf_counter()
      
      # Profiling
      if args.profile and args.profile_step_end == train_step:
        torch.cuda.cudart().cudaProfilerStop()

    # Write all metrics to file at the end of training
    with open(metrics_file, 'w') as f:
      for metrics in metrics_data:
        f.write(json.dumps(metrics) + '\n')

    logger.info(f"Training completed. Metrics saved to {metrics_file}")


  if backend is not None:
    with sdpa_kernel(backends=[backend]):
      training_loop()
  else:
    training_loop()

if __name__ == "__main__":
  init_logger()
  args = get_args()
  train(args)