#!/bin/bash
# Bash script to run coherence matcher on multiple JSON files
# Usage: ./run_coherence_eval.sh [directory|file1 file2 ...]

# Default settings
DIRECTORY="Results/cast"
DEVICE="cuda"
COMPARE_FLAG="--compare"
VERBOSE_FLAG="-v"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --directory|-d)
            DIRECTORY="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --no-compare)
            COMPARE_FLAG=""
            shift
            ;;
        --verbose|-v)
            VERBOSE_FLAG="-v"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options] [files or directory]"
            echo ""
            echo "Options:"
            echo "  --directory, -d DIR    Directory containing JSON files (default: Results/cast)"
            echo "  --device DEVICE        Device to use (default: cuda)"
            echo "  --no-compare          Skip baseline comparison"
            echo "  --verbose, -v         Verbose output"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 Results/cast/*.json"
            echo "  $0 -d Results/cast --device cuda:0"
            echo "  $0 --no-compare Results/cast/eval_cast_llama_sorrybench_20260223_005605.json"
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

# Get list of files
if [[ $# -gt 0 ]]; then
    # Use provided files
    FILES=("$@")
else
    # Use all JSON files in directory
    if [[ -d "$DIRECTORY" ]]; then
        FILES=("$DIRECTORY"/*.json)
    else
        echo "Error: Directory $DIRECTORY does not exist"
        exit 1
    fi
fi

# Check if files exist
if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No JSON files found"
    exit 1
fi

# Counter
TOTAL=${#FILES[@]}
CURRENT=0

echo "========================================"
echo "Coherence Evaluation Batch Runner"
echo "========================================"
echo "Files to process: $TOTAL"
echo "Device: $DEVICE"
echo "Compare mode: $([ -n "$COMPARE_FLAG" ] && echo 'Yes' || echo 'No')"
echo "========================================"
echo ""

# Loop through files
for FILE in "${FILES[@]}"; do
    # Skip if not a file
    [[ ! -f "$FILE" ]] && continue
    
    # Skip if already processed
    BASENAME=$(basename "$FILE")
    DIRNAME=$(dirname "$FILE")
    OUTPUT_FILE="${DIRNAME}/${BASENAME%.json}_coordinate.json"
    
    if [[ -f "$OUTPUT_FILE" ]]; then
        echo "[$((++CURRENT))/$TOTAL] Skipping $BASENAME (already processed)"
        continue
    fi
    
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL] Processing: $BASENAME"
    echo "    Output: $OUTPUT_FILE"
    
    # Run the coherence evaluator
    python behaviour_evaluator.py "$FILE" \
        --device "$DEVICE" \
        $COMPARE_FLAG \
        $VERBOSE_FLAG \
        --output "$OUTPUT_FILE"
    
    EXIT_CODE=$?
    
    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "    ✓ Success"
    else
        echo "    ✗ Failed (exit code: $EXIT_CODE)"
    fi
    
    echo ""
done

echo "========================================"
echo "Batch processing complete!"
echo "Processed: $CURRENT files"
echo "========================================"
