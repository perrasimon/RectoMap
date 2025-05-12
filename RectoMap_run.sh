#!/bin/bash

# Parse input (-i) and output (-o) arguments
while getopts "i:o:" opt; do
  case ${opt} in
    i )
      INPUT_PATH=$OPTARG
      ;;
    o )
      OUTPUT_BASE_PATH=$OPTARG
      ;;
    \? )
      echo "Usage: $0 -i /path/to/testing/images/directory -o /path/to/output/predictions/directory"
      exit 1
      ;;
  esac
done

# Check that both input and output paths are provided
if [ -z "$INPUT_PATH" ] || [ -z "$OUTPUT_BASE_PATH" ]; then
    echo "Error: Both input (-i) and output (-o) paths are required."
    echo "Usage: $0 -i /path/to/imagesTs -o /path/to/output_predictions"
    exit 1
fi

# Loop through all 5 folds and run predictions
for i in {0..4}; do
    OUTPUT_DIR="${OUTPUT_BASE_PATH}/predictions_fold${i}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "🔹 Running prediction for fold $i..."
    
    # Use different trainer for fold 1
    if [ "$i" -eq 1 ]; then
        TRAINER="nnUNetTrainer_UMambaBot_AUG_3d"
    else
        TRAINER="nnUNetTrainer_AUG_3d"
    fi

    nnUNetv2_predict -d Dataset009_RectalCancer -i "$INPUT_PATH" -o "$OUTPUT_DIR" \
        -f $i -tr $TRAINER -c 3d_fullres -p nnUNetPlans --save_probabilities \
        -chk checkpoint_best.pth
    
    echo "✅ Prediction for fold $i completed and saved to $OUTPUT_DIR"
    echo "------------------------------------------------------------"
done

echo "🚀 Starting STAPLE ensembling..."
python STAPLE_ensembling.py "$OUTPUT_BASE_PATH"


echo "🎉 All predictions completed successfully!"
