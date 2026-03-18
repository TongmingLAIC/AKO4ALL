#!/bin/bash
# setup.sh - Create isolated child environments for GPU kernel optimization
#
# Usage:
#   bash setup.sh --operator <name>                                 # local A100, from scratch
#   bash setup.sh --operator <name> --name "experiment_1"           # local A100 with label
#   bash setup.sh --operator <name> --backend modal                 # Modal B200, from scratch
#   bash setup.sh --operator <name> --mode existing --kernel /path/to/kernel.py
#   bash setup.sh --operator <name> --dataset /path/to/dataset      # custom dataset
#   bash setup.sh --operator <name> --prompt /path/to/prompt.md     # custom prompt
#   bash setup.sh --operator <name> --info /path/to/info.md         # custom info
#   bash setup.sh                                                   # list available operators

set -euo pipefail

# --- Defaults ---
MODE="scratch"
BACKEND="local"
KERNEL_PATH=""
LABEL=""
OPERATOR=""
DATASET_PATH=""
PROMPT_PATH=""
INFO_PATH=""

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --kernel)
            KERNEL_PATH="$2"
            shift 2
            ;;
        --name)
            LABEL="$2"
            shift 2
            ;;
        --operator)
            OPERATOR="$2"
            shift 2
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --prompt)
            PROMPT_PATH="$2"
            shift 2
            ;;
        --info)
            INFO_PATH="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            echo "Usage: bash setup.sh --operator <name> [--dataset <path>] [--mode scratch|existing] [--backend local|modal] [--kernel <path>] [--name <label>] [--prompt <path>] [--info <path>]"
            exit 1
            ;;
    esac
done

# --- Resolve paths ---
PARENT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$PARENT_DIR")"

# --- Resolve dataset path ---
if [[ -n "$DATASET_PATH" ]]; then
    FIB_DATASET_PATH="$DATASET_PATH"
else
    FIB_DATASET_PATH="${FIB_DATASET_PATH:-}"
fi
if [[ -z "$FIB_DATASET_PATH" ]]; then
    echo "Error: No dataset path specified."
    echo "Set FIB_DATASET_PATH or use --dataset to specify the path to the flashinfer-bench trace set."
    exit 1
fi
if [[ ! -d "$FIB_DATASET_PATH/definitions" ]]; then
    echo "Error: Dataset not found at $FIB_DATASET_PATH/definitions"
    echo "Set FIB_DATASET_PATH or use --dataset to specify the path to the flashinfer-bench trace set."
    exit 1
fi

# --- Resolve prompt and info paths ---
if [[ -z "$PROMPT_PATH" ]]; then
    PROMPT_PATH="$PARENT_DIR/templates/prompt.md"
fi
if [[ ! -f "$PROMPT_PATH" ]]; then
    echo "Error: Prompt file not found: $PROMPT_PATH"
    exit 1
fi

if [[ -z "$INFO_PATH" ]]; then
    INFO_PATH="$PARENT_DIR/templates/info.md"
fi
if [[ ! -f "$INFO_PATH" ]]; then
    echo "Error: Info file not found: $INFO_PATH"
    exit 1
fi

# --- List operators if none specified ---
if [[ -z "$OPERATOR" ]]; then
    echo "Available operators:"
    echo ""
    for def_file in "$FIB_DATASET_PATH"/definitions/*/*.json; do
        op_name="$(basename "$def_file" .json)"
        op_type="$(basename "$(dirname "$def_file")")"
        echo "  $op_name  ($op_type)"
    done
    echo ""
    echo "Usage: bash setup.sh --operator <name> [--dataset <path>] [--mode scratch|existing] [--backend local|modal] [--kernel <path>] [--name <label>] [--prompt <path>] [--info <path>]"
    exit 0
fi

# --- Auto-discover operator definition ---
DEFINITION_FILE=""
for def_file in "$FIB_DATASET_PATH"/definitions/*/"${OPERATOR}.json"; do
    if [[ -f "$def_file" ]]; then
        DEFINITION_FILE="$def_file"
        break
    fi
done

if [[ -z "$DEFINITION_FILE" ]]; then
    echo "Error: Operator '$OPERATOR' not found in $FIB_DATASET_PATH/definitions/"
    echo "Run without --operator to list available operators."
    exit 1
fi

# Find workloads file
OP_TYPE="$(basename "$(dirname "$DEFINITION_FILE")")"
WORKLOADS_FILE="$FIB_DATASET_PATH/workloads/${OP_TYPE}/${OPERATOR}.jsonl"
if [[ ! -f "$WORKLOADS_FILE" ]]; then
    echo "Error: Workloads file not found: $WORKLOADS_FILE"
    exit 1
fi

echo "Operator: $OPERATOR ($OP_TYPE)"
echo "Definition: $DEFINITION_FILE"
echo "Workloads: $WORKLOADS_FILE"
echo "Dataset: $FIB_DATASET_PATH"

# --- Validate ---
if [[ "$MODE" != "scratch" && "$MODE" != "existing" ]]; then
    echo "Error: --mode must be 'scratch' or 'existing' (got '$MODE')"
    exit 1
fi

if [[ "$BACKEND" != "local" && "$BACKEND" != "modal" ]]; then
    echo "Error: --backend must be 'local' or 'modal' (got '$BACKEND')"
    exit 1
fi

if [[ "$MODE" == "existing" ]]; then
    if [[ -z "$KERNEL_PATH" ]]; then
        echo "Error: --mode existing requires --kernel <path>"
        exit 1
    fi
    if [[ ! -f "$KERNEL_PATH" ]]; then
        echo "Error: Kernel file not found: $KERNEL_PATH"
        exit 1
    fi
    if ! grep -q 'def run(' "$KERNEL_PATH"; then
        echo "Error: Kernel file does not contain 'def run(': $KERNEL_PATH"
        exit 1
    fi
fi

# --- Generate template context ---
CONTEXT_FILE=$(mktemp)
trap 'rm -f "$CONTEXT_FILE"' EXIT

python3 "$PARENT_DIR/scripts/generate_context.py" \
    --definition "$DEFINITION_FILE" \
    --workloads "$WORKLOADS_FILE" \
    --output "$CONTEXT_FILE"

# shellcheck disable=SC1090
source "$CONTEXT_FILE"

# --- Auto-increment run number ---
MAX_NUM=0
shopt -s nullglob
for dir in "$BASE_DIR"/kernel-opt-agent-run-*/; do
    basename="$(basename "$dir")"
    # Extract the 3-digit number after "kernel-opt-agent-run-"
    num_part="${basename#kernel-opt-agent-run-}"
    num="${num_part%%-*}"  # strip everything after first dash (label)
    # Check if it's a valid number
    if [[ "$num" =~ ^[0-9]+$ ]]; then
        num=$((10#$num))  # force base-10
        if (( num > MAX_NUM )); then
            MAX_NUM=$num
        fi
    fi
done
shopt -u nullglob

NEXT_NUM=$(printf "%03d" $((MAX_NUM + 1)))

if [[ -n "$LABEL" ]]; then
    CHILD_NAME="kernel-opt-agent-run-${NEXT_NUM}-${LABEL}"
else
    CHILD_NAME="kernel-opt-agent-run-${NEXT_NUM}"
fi

CHILD_DIR="$BASE_DIR/$CHILD_NAME"

# --- Create child directory structure ---
mkdir -p "$CHILD_DIR/.claude"
mkdir -p "$CHILD_DIR/docs"
mkdir -p "$CHILD_DIR/scripts"
mkdir -p "$CHILD_DIR/solution/triton"

# --- Copy common files ---
cp "$PARENT_DIR/.gitignore" "$CHILD_DIR/.gitignore"
cp "$DEFINITION_FILE" "$CHILD_DIR/docs/definition.json"
cp "$WORKLOADS_FILE" "$CHILD_DIR/docs/workloads.jsonl"
cp "$PARENT_DIR/scripts/pack_solution.py" "$CHILD_DIR/scripts/pack_solution.py"

# --- Copy Info.md ---
cp "$INFO_PATH" "$CHILD_DIR/Info.md"

# --- Generate config.toml ---
cat > "$CHILD_DIR/config.toml" <<TOML
[solution]
name = "${OPERATOR}-reference"
definition = "${OPERATOR}"
author = "user"

[build]
language = "triton"
entry_point = "kernel.py::run"
destination_passing_style = false
TOML

# --- Copy backend-specific scripts ---
if [[ "$BACKEND" == "local" ]]; then
    cp "$PARENT_DIR/scripts/bench.sh" "$CHILD_DIR/scripts/bench.sh"
    cp "$PARENT_DIR/scripts/run_local.py" "$CHILD_DIR/scripts/run_local.py"
else
    cp "$PARENT_DIR/scripts/bench_modal.sh" "$CHILD_DIR/scripts/bench.sh"
    cp "$PARENT_DIR/scripts/run_modal.py" "$CHILD_DIR/scripts/run_modal.py"
fi

# --- Generate .claude/settings.local.json ---
cp "$PARENT_DIR/templates/settings/${BACKEND}.json" "$CHILD_DIR/.claude/settings.local.json"

# --- Copy kernel (and config.toml if colocated with kernel) ---
if [[ "$MODE" == "scratch" ]]; then
    # Extract reference kernel from definition.json
    python3 -c "
import json, sys
with open('$DEFINITION_FILE') as f:
    d = json.load(f)
ref = d.get('reference', '')
if not ref:
    print('Error: No reference field in definition.json', file=sys.stderr)
    sys.exit(1)
with open('$CHILD_DIR/solution/triton/kernel.py', 'w') as f:
    f.write(ref)
"
else
    cp "$KERNEL_PATH" "$CHILD_DIR/solution/triton/kernel.py"
    # If a config.toml exists next to the kernel (e.g. from a trajectory snapshot),
    # use it to preserve destination_passing_style and other settings
    KERNEL_DIR="$(dirname "$KERNEL_PATH")"
    if [[ -f "$KERNEL_DIR/config.toml" ]]; then
        cp "$KERNEL_DIR/config.toml" "$CHILD_DIR/config.toml"
        echo "Note: Using config.toml from $(realpath "$KERNEL_DIR/config.toml")"
    fi
fi

# --- Read Prompt.md content ---
PROMPT_CONTENT=$(cat "$PROMPT_PATH")

# --- Generate CLAUDE.md ---
GPU_SHORT=$([[ "$BACKEND" == "local" ]] && echo "A100 GPU" || echo "B200 GPU")
OBJECTIVE=$(cat "$PARENT_DIR/templates/claude-md/objective-${MODE}.md")
OBJECTIVE="${OBJECTIVE//\$\{GPU_SHORT\}/$GPU_SHORT}"
OBJECTIVE="${OBJECTIVE//\{\{OPERATOR_DESCRIPTION\}\}/$OPERATOR_DESCRIPTION}"
OBJECTIVE="${OBJECTIVE//\{\{OPERATOR\}\}/$OPERATOR}"
ENVIRONMENT=$(cat "$PARENT_DIR/templates/claude-md/environment-${BACKEND}.md")

# Build CLAUDE.md by replacing placeholders in common.md
# Use awk since sed struggles with multi-line replacements
awk \
    -v prompt="$PROMPT_CONTENT" \
    -v objective="$OBJECTIVE" \
    -v environment="$ENVIRONMENT" \
    -v operator="$OPERATOR" \
    -v operator_desc="$OPERATOR_DESCRIPTION" \
    -v num_workloads="$NUM_WORKLOADS" \
    -v shape_summary="$SHAPE_SUMMARY" \
    -v workload_summary="$WORKLOAD_SUMMARY" \
    '
    /\{\{PROMPT\}\}/ { print prompt; next }
    /\{\{OBJECTIVE\}\}/ { print objective; next }
    /\{\{ENVIRONMENT\}\}/ { print environment; next }
    /\{\{OPERATOR\}\}/ { gsub(/\{\{OPERATOR\}\}/, operator); print; next }
    /\{\{OPERATOR_DESCRIPTION\}\}/ { gsub(/\{\{OPERATOR_DESCRIPTION\}\}/, operator_desc); print; next }
    /\{\{NUM_WORKLOADS\}\}/ { gsub(/\{\{NUM_WORKLOADS\}\}/, num_workloads); print; next }
    /\{\{SHAPE_SUMMARY\}\}/ { print shape_summary; next }
    /\{\{WORKLOAD_SUMMARY\}\}/ { print workload_summary; next }
    { print }
' "$PARENT_DIR/templates/claude-md/common.md" > "$CHILD_DIR/CLAUDE.md"

# --- Git init ---
cd "$CHILD_DIR"
git init -q
git add -A
git commit -q -m "Initial commit (spawned from kernel-opt-agent, operator=$OPERATOR, mode=$MODE, backend=$BACKEND)"

# --- Summary ---
echo ""
echo "===== Child environment created ====="
echo "  Path:     $CHILD_DIR"
echo "  Operator: $OPERATOR"
echo "  Mode:     $MODE"
echo "  Backend:  $BACKEND"
echo "  Dataset:  $FIB_DATASET_PATH"
if [[ "$MODE" == "existing" ]]; then
    echo "  Kernel:   $KERNEL_PATH"
fi
echo ""
echo "Next steps:"
echo "  cd $CHILD_DIR"
echo ""
echo "  # Start your code agent, e.g.:"
echo "  claude"
echo ""
echo "  # Then send an optimization instruction, e.g.:"
echo "  # \"Read CLAUDE.md and Info.md, then begin optimizing the kernel.\""
echo ""
echo "To run benchmark manually:"
echo "  bash scripts/bench.sh"
echo "======================================="
