#!/bin/bash
################################################################################
# Generate YOLOE-26 proposals for HOI test set images
#
# Runs the proposal generation script inside the Ultralytics Docker container
# with GPU access. Generates proposals for SWIG and HICO test images.
#
# Prerequisites:
#   - Docker with NVIDIA GPU support
#   - hoi-dataset-curation repo at /media/shaun/workspace/hoi-dataset-curation/
#   - HOI dataset images at /media/shaun/workspace/hoi/dataset/
#
# Output:
#   /media/shaun/workspace/hoi-dataset-curation/output/test_proposals/
#   - SWIG: {action}_{id}.json files
#   - HICO: HICO_test2015_{id}.json files
#
# Runtime: ~30-60 min with GPU
#
# Usage:
#   bash generate_test_proposals.sh           # Generate both SWIG and HICO
#   bash generate_test_proposals.sh swig      # SWIG only
#   bash generate_test_proposals.sh hico      # HICO only
################################################################################

set -e

DATASET="${1:-both}"
CURATION_ROOT="/media/shaun/workspace/hoi-dataset-curation"
DATA_ROOT="/media/shaun/workspace/hoi/dataset"
OUTPUT_DIR="/app/output/test_proposals"
DOCKER_IMAGE="ultralytics/ultralytics:latest-nvidia-arm64"

echo "========================================================================"
echo "HOI Test Set Proposal Generation (YOLOE-26)"
echo "========================================================================"
echo "Curation root: $CURATION_ROOT"
echo "Data root:     $DATA_ROOT"
echo "Output dir:    ${CURATION_ROOT}/output/test_proposals/"
echo "Dataset:       $DATASET"
echo "========================================================================"
echo ""

# Verify Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Please install Docker with NVIDIA support."
    exit 1
fi

# Generate SWIG test proposals
run_swig() {
    echo "========================================================================"
    echo "Generating SWIG-HOI test proposals..."
    echo "  Scan dir: /data/swig_hoi/images_512"
    echo "  Output:   ${OUTPUT_DIR}"
    echo "========================================================================"
    docker run --rm --gpus all --ipc=host \
        -v "${CURATION_ROOT}:/app" \
        -v "${DATA_ROOT}:/data" \
        -w /app -e PYTHONPATH=/app \
        "${DOCKER_IMAGE}" \
        bash -c "pip install python-dotenv pillow tqdm -q && \
        python scripts/generate_proposals.py \
            --scan-dir /data/swig_hoi/images_512 \
            --output-dir ${OUTPUT_DIR} \
            --conf 0.3 \
            --max-proposals 50"
    echo "✓ SWIG proposals complete"
}

# Generate HICO test proposals
run_hico() {
    echo "========================================================================"
    echo "Generating HICO-DET test proposals..."
    echo "  Scan dir: /data/hico_20160224_det/images/test2015"
    echo "  Output:   ${OUTPUT_DIR}"
    echo "========================================================================"
    docker run --rm --gpus all --ipc=host \
        -v "${CURATION_ROOT}:/app" \
        -v "${DATA_ROOT}:/data" \
        -w /app -e PYTHONPATH=/app \
        "${DOCKER_IMAGE}" \
        bash -c "pip install python-dotenv pillow tqdm -q && \
        python scripts/generate_proposals.py \
            --scan-dir /data/hico_20160224_det/images/test2015 \
            --output-dir ${OUTPUT_DIR} \
            --conf 0.3 \
            --max-proposals 50"
    echo "✓ HICO proposals complete"
}

case "$DATASET" in
    swig)
        run_swig
        ;;
    hico)
        run_hico
        ;;
    both|*)
        run_swig
        run_hico
        ;;
esac

echo ""
echo "========================================================================"
echo "Proposal generation complete!"
echo "========================================================================"
echo ""
echo "Verifying output..."
PROPOSAL_DIR="${CURATION_ROOT}/output/test_proposals"
if [ -d "$PROPOSAL_DIR" ]; then
    TOTAL=$(ls -1 "$PROPOSAL_DIR"/*.json 2>/dev/null | wc -l)
    echo "  Total proposal files: $TOTAL"
    echo ""
    echo "Expected counts:"
    echo "  SWIG test: ~19,333 files (swig_hoi/images_512)"
    echo "  HICO test: ~20,028 files (hico test2015)"
    echo ""
    echo "Note: File count may be less than expected if some images had"
    echo "      no detections and were skipped, or if already cached."
else
    echo "  WARNING: Output directory not found: $PROPOSAL_DIR"
fi
echo ""
echo "Use these proposals with the SFT eval scripts:"
echo "  bash run_swig_ground_sft_eval.sh 0"
echo "  bash run_swig_action_sft_eval.sh 0"
echo "  bash run_hico_ground_sft_eval.sh 0"
echo "  bash run_hico_action_sft_eval.sh 0"
echo "========================================================================"
