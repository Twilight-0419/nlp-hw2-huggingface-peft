#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="$PWD/.cache/huggingface"
export WANDB_MODE=offline
export WANDB_DIR="$PWD"
export WANDB_CONFIG_DIR="$PWD/.wandb-config"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1

status_file="$PWD/progress.tsv"
mkdir -p outputs "$WANDB_CONFIG_DIR"
if [[ ! -s "$status_file" ]]; then
  printf 'timestamp\tphase\trun\tstatus\tdetail\n' > "$status_file"
fi

phase="$1"
shift
record() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$phase" "$1" "$2" "$3" >> "$status_file"
}

run_experiment() {
  local run="$1" model="$2" dataset="$3" seed="$4"
  shift 4
  local out="$PWD/outputs/$run"
  if [[ -f "$out/.complete" ]]; then
    return
  fi

  local available
  available="$(df -PB1 . | awk 'NR == 2 {print $4}')"
  if (( available < 3221225472 )); then
    record "$run" blocked "less than 3 GiB free"
    printf 'Insufficient free disk space for %s\n' "$run" >&2
    return 1
  fi

  mkdir -p "$out"
  local args=(
    --model_name "$model" --dataset "$dataset" --seed "$seed"
    --output_dir "$out" --run_name "$run" --do_train --do_eval
    --max_length 128 --per_device_train_batch_size 4
    --per_device_eval_batch_size 8 --gradient_accumulation_steps 4
    --bf16 --logging_steps 10 --report_to wandb --save_strategy no
    "$@"
  )
  local started="$SECONDS"
  {
    printf 'python -m hw2_huggingface.train '
    printf '%q ' "${args[@]}"
    printf '\n'
  } > "$out/console.log"
  record "$run" started "$model $dataset seed=$seed"
  if ! uv run --no-sync python \
      -m hw2_huggingface.train "${args[@]}" >> "$out/console.log" 2>&1; then
    record "$run" failed "training command"
    tail -n 30 "$out/console.log" >&2
    return 1
  fi

  if ! uv run --no-sync python - "$out" <<'PY'
from pathlib import Path
import json
import math
import sys

folder = Path(sys.argv[1])
eval_metrics = json.loads((folder / "eval_results.json").read_text())
train_metrics = json.loads((folder / "train_results.json").read_text())
state = json.loads((folder / "trainer_state.json").read_text())
assert state["global_step"] > 0
assert math.isfinite(train_metrics["train_loss"])
for name in ("eval_accuracy", "eval_micro_f1", "eval_macro_f1", "eval_weight_f1"):
    assert 0 <= eval_metrics[name] <= 1, (name, eval_metrics[name])
assert any(folder.glob("*.safetensors")), folder
PY
  then
    record "$run" failed "artifact validation"
    return 1
  fi
  touch "$out/.complete"
  record "$run" complete "$((SECONDS - started))s"
}

if [[ "$phase" == smoke ]]; then
  if (( $# )); then
    run_experiment "$@"
  else
    run_experiment smoke-bert bert-base-uncased agnews_sup 42 --max_steps 10 --learning_rate 2e-5 --logging_steps 1
    run_experiment smoke-bart facebook/bart-large agnews_sup 42 --max_steps 10 --learning_rate 2e-5 --logging_steps 1
    run_experiment smoke-qwen Qwen/Qwen1.5-0.5B agnews_sup 42 --max_steps 10 --learning_rate 2e-5 --logging_steps 1
    run_experiment smoke-bart-lora facebook/bart-large agnews_sup 42 \
      --max_steps 10 --learning_rate 1e-4 --peft lora --logging_steps 1
  fi
elif [[ "$phase" == formal ]]; then
  for model in bert-base-uncased facebook/bart-large Qwen/Qwen1.5-0.5B; do
    key="$(basename "$model")"
    for dataset in restaurant_sup acl_sup agnews_sup; do
      for seed in 42 43 44; do
        run_experiment "t2-$key-$dataset-full-s$seed" "$model" "$dataset" "$seed" \
          --num_train_epochs 2 --learning_rate 2e-5
      done
    done
  done
  for model in bert-base-uncased facebook/bart-large Qwen/Qwen1.5-0.5B; do
    key="$(basename "$model")"
    for method in lora adapter; do
      run_experiment "t3-$key-agnews-$method-s42" "$model" agnews_sup 42 \
        --peft "$method" --num_train_epochs 3 --learning_rate 1e-4
    done
  done
  for rank in 4 8 32; do
    run_experiment "t3-bert-agnews-lora-r$rank-s42" bert-base-uncased agnews_sup 42 \
      --peft lora --rank "$rank" --alpha "$((rank * 2))" \
      --num_train_epochs 3 --learning_rate 1e-4
  done
else
  printf 'Usage: %s smoke|formal\n' "$0" >&2
  exit 2
fi

record '-' phase_complete ok

