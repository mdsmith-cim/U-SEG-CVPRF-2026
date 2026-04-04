#!/bin/bash

# First time I ever managed to get a LLM to give me a good starting base
# gemma-3-27b-it-qat

# This is a quick script to run locally for timing evaluation
# It overrides the evaluators to not write the HDF5 files to disk as normal as they're not needed here

# Parse command line arguments using getopts
while getopts "o:" opt; do
  case $opt in
    o)
      OUTPUT_BASE_DIR="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      echo "Usage: $0 [-o output_base_dir] <file1> [<file2> ...]"
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      echo "Usage: $0 [-o output_base_dir] <file1> [<file2> ...]"
      exit 1
      ;;
  esac
done

shift $((OPTIND-1)) # Remove parsed options and their arguments from the argument list

# Check if at least one file is provided as an argument
if [ $# -lt 1 ]; then
  echo "Usage: $0 [-o output_base_dir] <file1> [<file2> ...]"
  exit 1
fi

# Optional output base directory
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-uncert_eval}" # Default to uncert_eval if not provided

# Create the output base directory if it doesn't exist
mkdir -p "$OUTPUT_BASE_DIR"

# Count the number of files provided
NUM_FILES=$#
echo "Number of files provided: $NUM_FILES"

# Iterate through each file and run train_net three times
for FILE in "$@"; do
  if [ ! -f "$FILE" ]; then
    echo "Error: '$FILE' is not a valid file."
    exit 1
  fi

  if [[ "$FILE" != *.yaml ]]; then
      echo "Error: '$FILE' is not a yaml file."
      exit 1
  fi

  CONFIG_FILE="$FILE"
  BASE_NAME=$(basename "$CONFIG_FILE" .yaml) # Get filename without extension

  for RUN in {1..3}; do
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${BASE_NAME}_run${RUN}"
    echo "Running eval for: $CONFIG_FILE with output directory: $OUTPUT_DIR"

    python train_net.py --num-gpus=1 --eval-only --config-file "$CONFIG_FILE" TEST.EVALUATORS '["PanopticQualityV3","SemSegEvaluatorV2"]' OUTPUT_DIR "$OUTPUT_DIR" SEED $RUN
    if [ $? -ne 0 ]; then
      echo "Error running eval for $CONFIG_FILE on run $RUN."
      exit 1
    fi
  done
done

echo "Eval completed."
exit 0
