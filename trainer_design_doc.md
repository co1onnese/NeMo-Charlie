1 — Executive summary

Build a repeatable, auditable pipeline to supervised-fine-tune (SFT) a causal LLM on stock-market analyst-style data (XML → HF dataset), evaluate both language-level and market-level performance, and provide a robust backtesting module to simulate historical trading performance. The architecture prioritizes reproducibility, low-memory fine-tuning (PEFT + QLoRA), strict time-based validation, and defensible evaluation metrics (classification + financial).

Phase 1 deliverables (finalized in this PRD):

Server prep & environment (Python venv)

Data ingestion: xml_to_jsonl.py and HF dataset creation

train_sft.py (TRL SFTTrainer + PEFT/QLoRA config) — production-ready skeleton

evaluate_sft.py — NLP + financial evaluation (per-sample CSV)

trading_backtest.py — backtest engine to simulate portfolio performance

Monitoring, reproducibility, and risk mitigation plan

2 — High-level architecture & dataflow

Raw data (XML files) — the XML format you provided (example included). 

example_input.xml

Ingestion (xml_to_jsonl.py) — robust parser → JSONL per thesis record (fields: ticker, as_of_date, instruction, input, output, metadata).

Dataset builder (convert_dataset.py) — JSONL → Hugging Face Dataset (Arrow), create train/validation/test splits by time.

Tokenization & preprocessing (tokenize_and_shard.py) — add special tokens, tokenize, produce shards.

SFT training (train_sft.py) — loads dataset, uses TRL SFTTrainer + PEFT (LoRA) + QLoRA (bitsandbytes 4-bit). Checkpointing + logging.

Evaluation (evaluate_sft.py) — generates predictions, extracts action tags, matches to historical price data, computes NLP and financial metrics.

Backtesting (trading_backtest.py) — simulates portfolio following model recommendations across evaluation period; produces equity curve, drawdown, Sharpe/Sortino, transaction-cost sensitivity.

Results & artifacts — CSVs, model checkpoints, tokenizer, dataset snapshot (Arrow), run logs (WandB / TensorBoard optional), runbook.

3 — Environment & server prep (venv-based)

Assume Ubuntu server with NVIDIA GPU and driver already installed. Use Python venv for reproducibility.

Commands to set up Python venv and install dependencies (example):

# create venv
python3 -m venv ~/sft-env
source ~/sft-env/bin/activate
pip install --upgrade pip setuptools wheel

# core libs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX  # choose correct CUDA wheel
pip install transformers accelerate datasets safetensors sentencepiece
pip install trl peft bitsandbytes
pip install huggingface_hub wandb scikit-learn pandas numpy matplotlib tqdm yfinance


Notes:

Replace cuXXX with the CUDA version compatible with your GPU / PyTorch wheel.

Test GPU availability:

python -c "import torch, bitsandbytes as bnb; print(torch.cuda.is_available()); print('bnb ok' if 'bitsandbytes' in globals() or bnb else 'bnb fail')"


If bitsandbytes wheel is incompatible, follow their build-from-source instructions (common on setups with non-standard CUDA).

4 — Core artifacts to create (files, modules)

Produce these files in the repo root (descriptions in parentheses):

data/

raw_xml/ (place raw XML files here)

hf_datasets/sft_dataset/ (Arrow dataset produced by convert_dataset.py)

src/

parsers/xml_to_jsonl.py (robust XML → JSONL parser)

data/convert_dataset.py (JSONL → HF Dataset, time-based splits, saving arrow)

data/tokenize_and_shard.py (tokenize with tokenizer.add_tokens, produce shards)

train/train_sft.py (SFTTrainer + PEFT/QLoRA training script)

eval/evaluate_sft.py (NLP + financial evaluation pipeline)

backtest/trading_backtest.py (backtesting engine & portfolio sim)

utils/validation.py (data validators, misc helpers)

utils/eval_utils.py (action parsing, price caching)

configs/

sft_config.yaml, eval_config.yaml, backtest_config.yaml (hyperparams, forward windows, fees)

runbook/

README.md (how to run steps, environment commands), checklist.md

5 — Data ingestion & preprocessing — detailed
5.1 XML → JSONL (xml_to_jsonl.py) — requirements

Parse each <thesis> tag; extract:

ticker (from parent attribute or tag)

as_of_date (normalize to ISO YYYY-MM-DD)

structured fields: <reasoning>,<support>,<action>, plus any indicators

metadata: generated-by, version, file provenance

Normalize nan, quoted numeric strings, and missing values → null or ""

For fields that are lists or multiple <thesis> per file, emit one JSON object per thesis

Add uid per record (ticker + as_of_date + running index)

Save as newline-delimited JSONL: one JSON object per line

Example JSONL skeleton:

{
  "uid": "TSLA|2025-07-01|0",
  "ticker": "TSLA",
  "as_of_date": "2025-07-01",
  "instruction": "Given this market snapshot, produce reasoning and action.",
  "input": "RSI(14)=nan, SMA(20)=...",
  "output": "<reasoning>...</reasoning><support>...</support><action>BUY</action>",
  "metadata": { "file": "raw_xml/file1.xml", "generated_by": "modelX-v1" }
}

5.2 HF dataset construction (convert_dataset.py)

Load JSONL → Hugging Face Dataset

Enforce time-based splits: sort by as_of_date, choose cutoffs for train / validation / test (e.g., last 6 months → test)

Save processed dataset with dataset.save_to_disk("data/hf_datasets/sft_dataset")

5.3 Tokenization & special tokens (tokenize_and_shard.py)

Add tokens to tokenizer:

XML tags: <reasoning>, </reasoning>, <support>, </support>, <action>, </action>

Action tokens: <STRONG_BUY>, <BUY>, <HOLD>, <SELL>, <STRONG_SELL>

Add any indicator tokens you want to treat specially if needed

Tokenize prompts and targets, create input_ids, attention_mask, and labels where labels = -100 for prefix tokens and real token ids for target tokens (target token alignment strategy described in training section)

Save tokenized Arrow shards to disk

6 — SFT training (train_sft.py) — full plan (summary)
6.1 Design decisions

Use TRL SFTTrainer with PEFT LoRA adapters (peft) and QLoRA (bitsandbytes 4-bit) for memory-efficient tuning of large models.

Keep base model frozen; train LoRA weights only.

Keep tags & action tokens in labels to let model learn structure.

Use device_map="auto" for multi-GPU or single GPU; use accelerate launch if multi-GPU.

6.2 Key steps in the script

Load tokenizer (add tokens if missing)

Load Arrow dataset (train/validation) and the tokenized shards

Build tokenized inputs with labels masked for prefix

Configure BitsAndBytesConfig and load pretrained model with quantization_config if using 4-bit

Run prepare_model_for_kbit_training and wrap with get_peft_model(LoraConfig(...))

Instantiate SFTTrainer with SFTConfig and train

Save adapters & tokenizer; optionally push to HF Hub

6.3 Suggested hyperparameters

max_length: 2048 (or 4096 if model supports)

per_device_train_batch_size: 1–4 (use grad accumulation)

gradient_accumulation_steps: 8–32

learning_rate: 1e-4 — 5e-4

lora_r: 8–32; lora_alpha: 16–32; lora_dropout: 0.05

num_train_epochs: 1–5 (monitor validation loss and action accuracy)

save_steps: 2000 (adjust by dataset size)

fp16: True if supported

Smoke-run first: run with small num_train_steps=10 and check outputs.

7 — Evaluation approach (deep-dive)

We split evaluation into language-level and market-level metrics.

7.1 Language-level (NLP) evaluation

Loss / Perplexity on validation split

Action classification: extract <action> from generated output and compare to gt_action. Report:

Accuracy, Precision/Recall/F1 per class, Confusion matrix

Reasoning quality: BLEU/ROUGE or embedding-based cosine similarity between pred_reasoning and gt_reasoning (optional)

Tag-structure validity: fraction of outputs that contain properly-closed tags (<reasoning>... </action>) — enforce during training by tokenizing tags as single tokens.

7.2 Market-level (financial) evaluation

Per-sample realized returns: For each prediction with as_of_date = t and ticker = T, compute forward return over windows Δ ∈ {1d, 5d, 10d, 30d}:

𝑅
Δ
=
𝑃
𝑡
+
Δ
−
𝑃
𝑡
𝑃
𝑡
R
Δ
	​

=
P
t
	​

P
t+Δ
	​

−P
t
	​

	​


Directional correctness: whether sign(R_Δ) aligns with predicted action polarity (mapping: BUY → +, SELL → -, HOLD → 0).

Hit rate (profitability): fraction of predicted BUYs that achieved R_Δ > 0 (or > threshold)

Average return per action: mean(R_Δ | predicted_action)

Portfolio-level metrics (from backtest): cumulative return, annualized Sharpe, Sortino, max drawdown, volatility, turnover

Transaction costs & slippage: model sensitivity to costs; compute net returns after fees (configurable in backtest).

7.3 Temporal integrity & dataset split

Train / Val / Test must be time-based (chronological). Example:

Train: all records up to 2023-12-31

Validation: 2024-01-01 → 2024-06-30

Test: 2024-07-01 → 2024-12-31

Avoid random splits which leak future info.

7.4 Mapping predicted action → trading rule

Map actions to position rules:

STRONG_BUY → open/size long position (weight 1.0)

BUY → open long (weight 0.5)

HOLD → no action / keep existing position

SELL → close long / open short (if shorts allowed)

STRONG_SELL → strong short

The backtest will have configurable rules for position sizing and allowed actions.

8 — Implementation: evaluate_sft.py (summary & responsibilities)

Script responsibilities:

Load model & tokenizer (fined-tuned checkpoint)

For each example in evaluation split:

Build prompt (instruction + input)

Generate deterministic completion (temperature=0, do_sample=False) or calibrated sampling if desired

Extract <action> via regex/robust parser (use utils/eval_utils.py)

Store pred_action, pred_reasoning, full completion

Retrieve price history for ticker and as_of_date (use cached CSV or yfinance if allowed)

Compute R_Δ for configured Δ windows

Produce:

results/eval_results.csv (per-sample)

Aggregate metrics: action classification report, hit rates per Δ, mean returns per action

Plots: return histograms, confusion matrix, equity curve snapshot (optional)

Important code details:

Use a price cache to avoid repeated API calls.

For mapping as_of_date to trading-day index, use the first available trading day at or after as_of_date.

Treat splits where t+Δ exceeds available data as NaN and exclude for certain metrics.

9 — Optional deeper backtesting (trading_backtest.py)

A robust backtester should support:

Position sizing rules (fixed, volatility-targeting, Kelly fraction)

Execution assumptions: entry at next-day open/close, slippage, spread

Transaction costs: fixed per-trade or basis-points

Leverage & margin rules (optional)

Portfolio construction:

Single-ticker positions vs multi-ticker portfolio

Rebalancing frequency (daily decisions are natural here)

Risk controls: max drawdown stop-loss, max position exposure

Metrics: cumulative return, annualized return, annualized volatility, Sharpe, Sortino, max drawdown, turnover, hit rate

Event-level simulation: apply each model prediction chronologically, maintain portfolio state (cash + holdings), produce equity curve

Backtest algorithm (simplified)

Initialize cash (e.g., $1,000,000), empty portfolio

For each day in chronological order:

Collect all model predictions for that day (and their tickers)

For each prediction, determine target position using mapping (BUY → go long X% of portfolio)

Execute trades at defined execution price (next_open or close) with fees/slippage

Update portfolio mark-to-market using daily closes

Track metrics at each day and at the end

Edge cases

Conflicting signals for same ticker same day — choose last prediction, aggregate, or average signals.

Non-tradable days (holidays) — use next trading day.

Corporate actions — use adjusted prices.

10 — Config & reproducibility

Create configs/ with YAML files to capture all hyperparameters & settings:

configs/sft_config.yaml (example keys):

base_model: decapoda-research/llama-7b-hf
max_length: 2048
train_batch_size: 2
gradient_accumulation_steps: 16
learning_rate: 2e-4
num_train_epochs: 3
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
load_in_4bit: true
bnb_quant_type: nf4
seed: 42


configs/eval_config.yaml:

forward_windows: [1, 5, 10, 30]
price_source: local_csv  # or yfinance
price_cache_path: data/price_cache.parquet
results_csv: results/eval_results.csv


configs/backtest_config.yaml:

initial_cash: 1000000
transaction_cost_bps: 5
slippage_bps: 10
entry_price: next_open
position_sizing: fixed_pct
fixed_pct_value: 0.02
allow_shorts: false


Always check in the config files and the exact pip dependency versions (e.g., requirements.txt) alongside the raw dataset sample (seeded) so runs are reproducible.

11 — Logging, monitoring, and artifacts

Logging: use Python logging for scripts; configure per-run logs to logs/<run_id>.log.

Training metrics: integrate WandB (wandb.init(project=...)) or use tensorboard for scalars and generated examples snapshots.

Artifacts:

Model checkpoints + tokenizer (checkpoints/sft-run/)

HF dataset snapshot (data/hf_datasets/sft_dataset/)

Evaluation CSVs & plots (results/)

Backtest results + equity curve images (backtests/)

A manifest.json per run recording: dataset checksum, model hash, git commit, config file, seed, timestamp

12 — Risks, failure modes, & mitigations

Driver/CUDA/bitsandbytes incompatibility

Mitigation: test bitsandbytes import on the server before long runs; have fallback to non-quantized (load_in_4bit=False) training.

Data leakage / time-split mistakes

Mitigation: enforce time-based splits in convert_dataset.py. Add unit tests verifying max(as_of_date) in train < min(as_of_date) in validation/test.

Model overfitting or degenerate outputs (e.g., always HOLD)

Mitigation: track class-wise metrics and per-class loss; use class-balancing or data augmentation if necessary; review samples during training.

Reward hacking in later RL phase

Mitigation: ensure evaluation/backtest uses robust composite reward (P&L + risk penalty + plausibility checks).

Market simulation mismatch

Mitigation: treat backtest as a diagnostic; simulate execution costs & slippage conservatively.

Regulatory/legal liability (financial advice)

Mitigation: add disclaimers, restrict model usage, follow legal review for deployment; require human-in-the-loop.

13 — Checklists
Pre-training checklist

 Raw XML files in data/raw_xml/

 xml_to_jsonl.py runs successfully on a sample file → JSONL

 convert_dataset.py produces time-based train/val/test Arrow dataset

 Tokenizer updated with special tokens; model.resize_token_embeddings saved

 venv active and packages installed; torch.cuda.is_available() is True

 Smoke-run of train_sft.py (10 steps) completes

Pre-evaluation checklist

 Model checkpoint saved and tokenizer in checkpoints/sft-run/

 Price data available & cached for evaluation period

 evaluate_sft.py runs on the test split and produces results/eval_results.csv

 Confusion matrix + per-class metrics reviewed

Backtest checklist

 trading_backtest.py configured and run with conservative costs

 Equity curve, Sharpe, drawdown computed and stored

 Sensitivity checks for transaction cost & slippage

14 — Example run commands (venv)

Create venv and install:

python3 -m venv ~/sft-env
source ~/sft-env/bin/activate
pip install -r requirements.txt


Ingest & prepare dataset:

python src/parsers/xml_to_jsonl.py --input_dir data/raw_xml --out data/jsonl/all.jsonl
python src/data/convert_dataset.py --jsonl data/jsonl/all.jsonl --out_dir data/hf_datasets/sft_dataset --test_split_days 180
python src/data/tokenize_and_shard.py --dataset_dir data/hf_datasets/sft_dataset --tokenizer MODEL-ID --out data/hf_datasets/tokenized_shards


Train (smoke / full):

# smoke run
python src/train/train_sft.py --config configs/sft_config.yaml --max_steps 10

# full run
python src/train/train_sft.py --config configs/sft_config.yaml


Evaluate:

python src/eval/evaluate_sft.py --model_dir checkpoints/sft-run --dataset_dir data/hf_datasets/sft_dataset --out results/eval_results.csv


Backtest:

python src/backtest/trading_backtest.py --eval_csv results/eval_results.csv --config configs/backtest_config.yaml --out backtests/baseline.json

15 — Deliverables in draft format:

✅ xml_to_jsonl.py tailored to your uploaded example XML (I have the sample). 

example_input.xml

✅ train_sft.py refined to your chosen base model and exact tokenizer fields (I provided a generic one earlier).

✅ evaluate_sft.py (complete, as described above) and utils/eval_utils.py for price caching and action parsing.

✅ trading_backtest.py implementing the backtest algorithm and sample configs for transaction costs / position sizing.

✅ configs/ and requirements.txt for the venv.
