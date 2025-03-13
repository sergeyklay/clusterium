#!/bin/bash
# Script to perform manual testing of clusx cluster & clusx evaluate commands
# This script will:
# 1. Load test parameters from a profile file
# 2. Execute clusx cluster with these parameters
# 3. Move output files to organized folders
# 4. Create JSON with performance metrics
# 5. Create readme.txt with run information

set -e  # Exit on error

# Display usage information
usage() {
    echo "Usage: $0 --input <input_file> --profile <profile_file> [--column <column_name>] [--random-seed <seed>] [--output <output_basename>]"
    echo ""
    echo "Arguments:"
    echo "  --input        Path to the input CSV file (required)"
    echo "  --profile      Path to the profile file with test parameters (required)"
    echo "  --column       Column name to use for clustering (default: 'question')"
    echo "  --random-seed  Random seed for reproducibility (default: random value)"
    echo "  --output       Base name for output files (default: 'clusters_output')"
    echo "  --output-dir   Directory to save output files (default: 'output')"
    echo "  --batch-name   Name of the batch (default: current date and time)"
    exit 1
}

# Parse command line arguments
INPUT_FILE=""
PROFILE_FILE=""
COLUMN="question"
RANDOM_SEED=$RANDOM  # Default to a random value
OUTPUT_BASENAME="clusters_output"
OUTPUT_DIR="output"
BATCH_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --profile)
            PROFILE_FILE="$2"
            shift 2
            ;;
        --column)
            COLUMN="$2"
            shift 2
            ;;
        --random-seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        --output)
            OUTPUT_BASENAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-name)
            BATCH_NAME="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if input file is provided
if [ -z "$INPUT_FILE" ]; then
    echo "Error: Input file is required"
    usage
fi

# Check if profile file is provided
if [ -z "$PROFILE_FILE" ]; then
    echo "Error: Profile file is required"
    usage
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' does not exist"
    exit 1
fi

# Check if profile file exists
if [ ! -f "$PROFILE_FILE" ]; then
    echo "Error: Profile file '$PROFILE_FILE' does not exist"
    exit 1
fi

# Get current date and time for documentation and folder naming
CURRENT_DATE=$(date +"%Y-%m-%d")
CURRENT_DATETIME=$(date +"%Y%m%d-%H%M%S")

# If batch name is not provided, use the current date and time
if [ -z "$BATCH_NAME" ]; then
    BATCH_NAME="${CURRENT_DATETIME}"
fi

# Create batch directory structure
BATCH_DIR="${OUTPUT_DIR}/batch/${BATCH_NAME}"
mkdir -p "${BATCH_DIR}"

# Load test cases from profile file
# Strip empty lines and comments (lines starting with #)
declare -A test_cases
test_num=1

while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and comments
    if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    # Store non-empty, non-comment lines as test cases
    test_cases[$test_num]="$line"
    ((test_num++))
done < <(grep -v '^\s*$\|^\s*#' "$PROFILE_FILE")

# Get the total number of test cases
total_tests=${#test_cases[@]}

if [ $total_tests -eq 0 ]; then
    echo "Error: No valid test cases found in profile file"
    exit 1
fi

echo "Starting test run with input file: $INPUT_FILE"
echo "Profile file: $PROFILE_FILE"
echo "Column: $COLUMN"
echo "Random seed: $RANDOM_SEED"
echo "Output basename: $OUTPUT_BASENAME"
echo "Output directory: $OUTPUT_DIR"
echo "Batch name: $BATCH_NAME"
echo "Number of test cases: $total_tests"
echo ""

# Create temporary directory for intermediate files
TEMP_DIR="${OUTPUT_DIR}/temp"
mkdir -p "${TEMP_DIR}"

# Run each test case
for test_num in $(seq 1 $total_tests); do
    echo "======================================================="
    echo "Running Test $test_num"
    echo "Parameters: ${test_cases[$test_num]}"
    echo "======================================================="

    # Extract parameters
    PARAMS=${test_cases[$test_num]}

    # Create run directory with proper structure
    RUN_DIR="${BATCH_DIR}/${test_num}"
    mkdir -p "${RUN_DIR}"

    # Dynamically build output file names
    DP_CLUSTERS_CSV="${TEMP_DIR}/${OUTPUT_BASENAME}_dp.csv"
    PYP_CLUSTERS_CSV="${TEMP_DIR}/${OUTPUT_BASENAME}_pyp.csv"
    DP_CLUSTERS_JSON="${TEMP_DIR}/${OUTPUT_BASENAME}_dp.json"
    PYP_CLUSTERS_JSON="${TEMP_DIR}/${OUTPUT_BASENAME}_pyp.json"

    # Run clusx cluster command
    echo "Executing: clusx cluster --input $INPUT_FILE --column $COLUMN $PARAMS --random-seed $RANDOM_SEED --output $OUTPUT_BASENAME --output-dir $TEMP_DIR"
    clusx cluster --input "$INPUT_FILE" --column "$COLUMN" $PARAMS --random-seed $RANDOM_SEED --output "$OUTPUT_BASENAME.csv" --output-dir "$TEMP_DIR"

    # Run clusx evaluate command with proper parameters
    echo "Executing: clusx evaluate --input $INPUT_FILE --column $COLUMN --dp-clusters $DP_CLUSTERS_CSV --pyp-clusters $PYP_CLUSTERS_CSV --random-seed $RANDOM_SEED --plot --output-dir $TEMP_DIR"
    clusx evaluate --input "$INPUT_FILE" --column "$COLUMN" --dp-clusters "$DP_CLUSTERS_CSV" --pyp-clusters "$PYP_CLUSTERS_CSV" --random-seed $RANDOM_SEED --plot --output-dir "$TEMP_DIR"

    # Move output files to run directory
    echo "Moving output files to $RUN_DIR"
    find "${TEMP_DIR}" -type f -name "*.json" -o -name "*.csv" -o -name "*.png" | xargs -I{} mv {} "${RUN_DIR}/"

    # Extract stats and create performance metrics JSON
    echo "Extracting performance metrics..."

    # Check if evaluation_report.json exists
    if [ -f "${RUN_DIR}/evaluation_report.json" ]; then
        # Extract Dirichlet stats
        d_num_clusters=$(jq '.Dirichlet.cluster_stats.num_clusters' "${RUN_DIR}/evaluation_report.json")

        # Extract cluster size distribution for Dirichlet
        d_size_1=$(jq '.Dirichlet.cluster_stats.cluster_sizes | to_entries | map(select(.key == "1")) | .[0].value // 0' "${RUN_DIR}/evaluation_report.json")
        d_size_2_5=$(jq '.Dirichlet.cluster_stats.cluster_sizes | to_entries | map(select(.key == "2" or .key == "3" or .key == "4" or .key == "5")) | map(.value) | add // 0' "${RUN_DIR}/evaluation_report.json")
        d_size_6plus=$(jq '.Dirichlet.cluster_stats.cluster_sizes | to_entries | map(select(.key | tonumber >= 6)) | map(.value) | add // 0' "${RUN_DIR}/evaluation_report.json")

        # Extract powerlaw stats for Dirichlet
        d_powerlaw_alpha=$(jq '.Dirichlet.metrics.powerlaw.alpha // 0' "${RUN_DIR}/evaluation_report.json")
        d_is_powerlaw=$(jq '.Dirichlet.metrics.powerlaw.is_powerlaw // false' "${RUN_DIR}/evaluation_report.json")

        # Extract similarity stats for Dirichlet
        d_silhouette=$(jq '.Dirichlet.metrics.silhouette_score // 0' "${RUN_DIR}/evaluation_report.json")
        d_intra_sim=$(jq '.Dirichlet.metrics.similarity.intra_cluster_similarity // 0' "${RUN_DIR}/evaluation_report.json")
        d_inter_sim=$(jq '.Dirichlet.metrics.similarity.inter_cluster_similarity // 0' "${RUN_DIR}/evaluation_report.json")
        d_silhouette_like=$(jq '.Dirichlet.metrics.similarity.silhouette_like_score // 0' "${RUN_DIR}/evaluation_report.json")

        # Extract Pitman-Yor stats
        py_num_clusters=$(jq '."Pitman-Yor".cluster_stats.num_clusters' "${RUN_DIR}/evaluation_report.json")

        # Extract cluster size distribution for Pitman-Yor
        py_size_1=$(jq '."Pitman-Yor".cluster_stats.cluster_sizes | to_entries | map(select(.key == "1")) | .[0].value // 0' "${RUN_DIR}/evaluation_report.json")
        py_size_2_5=$(jq '."Pitman-Yor".cluster_stats.cluster_sizes | to_entries | map(select(.key == "2" or .key == "3" or .key == "4" or .key == "5")) | map(.value) | add // 0' "${RUN_DIR}/evaluation_report.json")
        py_size_6plus=$(jq '."Pitman-Yor".cluster_stats.cluster_sizes | to_entries | map(select(.key | tonumber >= 6)) | map(.value) | add // 0' "${RUN_DIR}/evaluation_report.json")

        # Extract powerlaw stats for Pitman-Yor
        py_powerlaw_alpha=$(jq '."Pitman-Yor".metrics.powerlaw.alpha // 0' "${RUN_DIR}/evaluation_report.json")
        py_is_powerlaw=$(jq '."Pitman-Yor".metrics.powerlaw.is_powerlaw // false' "${RUN_DIR}/evaluation_report.json")

        # Extract similarity stats for Pitman-Yor
        py_silhouette=$(jq '."Pitman-Yor".metrics.silhouette_score // 0' "${RUN_DIR}/evaluation_report.json")
        py_intra_sim=$(jq '."Pitman-Yor".metrics.similarity.intra_cluster_similarity // 0' "${RUN_DIR}/evaluation_report.json")
        py_inter_sim=$(jq '."Pitman-Yor".metrics.similarity.inter_cluster_similarity // 0' "${RUN_DIR}/evaluation_report.json")
        py_silhouette_like=$(jq '."Pitman-Yor".metrics.similarity.silhouette_like_score // 0' "${RUN_DIR}/evaluation_report.json")

        # Create performance metrics JSON
        cat > "${RUN_DIR}/performance_metrics.json" << EOF
{
  "Dirichlet": {
    "num_clusters": $d_num_clusters,
    "cluster_size_distribution": {"1": $d_size_1, "2-5": $d_size_2_5, "6+": $d_size_6plus},
    "powerlaw": {"alpha": $d_powerlaw_alpha, "is_powerlaw": $d_is_powerlaw},
    "silhouette_score": $d_silhouette,
    "similarity": {
      "intra": $d_intra_sim,
      "inter": $d_inter_sim,
      "silhouette_like": $d_silhouette_like
    }
  },
  "Pitman-Yor": {
    "num_clusters": $py_num_clusters,
    "cluster_size_distribution": {"1": $py_size_1, "2-5": $py_size_2_5, "6+": $py_size_6plus},
    "powerlaw": {"alpha": $py_powerlaw_alpha, "is_powerlaw": $py_is_powerlaw},
    "silhouette_score": $py_silhouette,
    "similarity": {
      "intra": $py_intra_sim,
      "inter": $py_inter_sim,
      "silhouette_like": $py_silhouette_like
    }
  }
}
EOF
    else
        echo "Warning: evaluation_report.json not found in ${RUN_DIR}"
        echo "{}" > "${RUN_DIR}/performance_metrics.json"
    fi

    # Create readme.txt with run information and performance metrics
    cat > "${RUN_DIR}/readme.txt" << EOF
Batch: $BATCH_NAME
Test Run: $test_num
Date: $CURRENT_DATE
Input File: $INPUT_FILE
Profile File: $PROFILE_FILE
Column: $COLUMN
Random Seed: $RANDOM_SEED
Output Basename: $OUTPUT_BASENAME

Parameters: ${test_cases[$test_num]}

Cluster Command:
clusx cluster --input $INPUT_FILE --column $COLUMN $PARAMS --random-seed $RANDOM_SEED --output $OUTPUT_BASENAME.csv --output-dir $TEMP_DIR

Evaluation Command:
clusx evaluate --input $INPUT_FILE --column $COLUMN --dp-clusters $DP_CLUSTERS_CSV --pyp-clusters $PYP_CLUSTERS_CSV --random-seed $RANDOM_SEED --plot --output-dir $TEMP_DIR

Performance Metrics:
$(cat "${RUN_DIR}/performance_metrics.json" | jq -r .)
EOF

    echo "Test $test_num completed. Results saved to ${RUN_DIR}"
    echo ""
done

# Clean up temporary directory
rm -rf "${TEMP_DIR}"

# Create a summary report
echo "Creating summary report..."
SUMMARY_FILE="${BATCH_DIR}/summary.md"
SUMMARY_CSV="${BATCH_DIR}/summary.csv"

cat > "${SUMMARY_FILE}" << EOF
# Clusx Test Summary
Batch: $BATCH_NAME
Date: $CURRENT_DATE
Input File: $INPUT_FILE
Profile File: $PROFILE_FILE
Column: $COLUMN
Random Seed: $RANDOM_SEED
Output Basename: $OUTPUT_BASENAME

## Test Cases

EOF

# Create CSV header
echo "test_num,model,dp_alpha,pyp_alpha,pyp_sigma,variance,num_clusters,cluster_size_1,cluster_size_2_5,cluster_size_6plus,powerlaw_alpha,is_powerlaw,silhouette_score,intra_similarity,inter_similarity,silhouette_like" > "${SUMMARY_CSV}"

for test_num in $(seq 1 $total_tests); do
    RUN_DIR="${BATCH_DIR}/${test_num}"

    # Extract parameters from test case
    dp_alpha=$(echo "${test_cases[$test_num]}" | grep -o "\--dp-alpha [0-9.]*" | awk '{print $2}')
    pyp_alpha=$(echo "${test_cases[$test_num]}" | grep -o "\--pyp-alpha [0-9.]*" | awk '{print $2}')
    pyp_sigma=$(echo "${test_cases[$test_num]}" | grep -o "\--pyp-sigma [0-9.]*" | awk '{print $2}')
    variance=$(echo "${test_cases[$test_num]}" | grep -o "\--variance [0-9.]*" | awk '{print $2}')

    # Add to markdown summary
    cat >> "${SUMMARY_FILE}" << EOF
### Test $test_num
Parameters: ${test_cases[$test_num]}

\`\`\`json
$(cat "${RUN_DIR}/performance_metrics.json")
\`\`\`

EOF

    # Add to CSV summary if performance_metrics.json exists
    if [ -f "${RUN_DIR}/performance_metrics.json" ]; then
        # Dirichlet model
        d_num_clusters=$(jq '.Dirichlet.num_clusters' "${RUN_DIR}/performance_metrics.json")
        d_size_1=$(jq '.Dirichlet.cluster_size_distribution."1"' "${RUN_DIR}/performance_metrics.json")
        d_size_2_5=$(jq '.Dirichlet.cluster_size_distribution."2-5"' "${RUN_DIR}/performance_metrics.json")
        d_size_6plus=$(jq '.Dirichlet.cluster_size_distribution."6+"' "${RUN_DIR}/performance_metrics.json")
        d_powerlaw_alpha=$(jq '.Dirichlet.powerlaw.alpha' "${RUN_DIR}/performance_metrics.json")
        d_is_powerlaw=$(jq '.Dirichlet.powerlaw.is_powerlaw' "${RUN_DIR}/performance_metrics.json")
        d_silhouette=$(jq '.Dirichlet.silhouette_score' "${RUN_DIR}/performance_metrics.json")
        d_intra_sim=$(jq '.Dirichlet.similarity.intra' "${RUN_DIR}/performance_metrics.json")
        d_inter_sim=$(jq '.Dirichlet.similarity.inter' "${RUN_DIR}/performance_metrics.json")
        d_silhouette_like=$(jq '.Dirichlet.similarity.silhouette_like' "${RUN_DIR}/performance_metrics.json")

        # Pitman-Yor model
        py_num_clusters=$(jq '."Pitman-Yor".num_clusters' "${RUN_DIR}/performance_metrics.json")
        py_size_1=$(jq '."Pitman-Yor".cluster_size_distribution."1"' "${RUN_DIR}/performance_metrics.json")
        py_size_2_5=$(jq '."Pitman-Yor".cluster_size_distribution."2-5"' "${RUN_DIR}/performance_metrics.json")
        py_size_6plus=$(jq '."Pitman-Yor".cluster_size_distribution."6+"' "${RUN_DIR}/performance_metrics.json")
        py_powerlaw_alpha=$(jq '."Pitman-Yor".powerlaw.alpha' "${RUN_DIR}/performance_metrics.json")
        py_is_powerlaw=$(jq '."Pitman-Yor".powerlaw.is_powerlaw' "${RUN_DIR}/performance_metrics.json")
        py_silhouette=$(jq '."Pitman-Yor".silhouette_score' "${RUN_DIR}/performance_metrics.json")
        py_intra_sim=$(jq '."Pitman-Yor".similarity.intra' "${RUN_DIR}/performance_metrics.json")
        py_inter_sim=$(jq '."Pitman-Yor".similarity.inter' "${RUN_DIR}/performance_metrics.json")
        py_silhouette_like=$(jq '."Pitman-Yor".similarity.silhouette_like' "${RUN_DIR}/performance_metrics.json")

        # Add Dirichlet row to CSV
        echo "$test_num,Dirichlet,$dp_alpha,$pyp_alpha,$pyp_sigma,$variance,$d_num_clusters,$d_size_1,$d_size_2_5,$d_size_6plus,$d_powerlaw_alpha,$d_is_powerlaw,$d_silhouette,$d_intra_sim,$d_inter_sim,$d_silhouette_like" >> "${SUMMARY_CSV}"

        # Add Pitman-Yor row to CSV
        echo "$test_num,Pitman-Yor,$dp_alpha,$pyp_alpha,$pyp_sigma,$variance,$py_num_clusters,$py_size_1,$py_size_2_5,$py_size_6plus,$py_powerlaw_alpha,$py_is_powerlaw,$py_silhouette,$py_intra_sim,$py_inter_sim,$py_silhouette_like" >> "${SUMMARY_CSV}"
    fi
done

echo "Testing completed. Summary available at ${SUMMARY_FILE} and ${SUMMARY_CSV}"
echo "Results stored in ${BATCH_DIR}"
