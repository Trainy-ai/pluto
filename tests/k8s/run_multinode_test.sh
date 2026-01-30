#!/bin/bash
# K8s multinode E2E test runner using konduktor CLI
#
# This script demonstrates launching a distributed training job with konduktor
# where all nodes share the same PLUTO_RUN_ID and log to the same Pluto run.
#
# Based on the konduktor CI pattern - requires:
# - JobSet controller
# - Kueue controller
# - LocalQueue 'user-queue'
#
# Prerequisites:
# - docker
# - kind (https://kind.sigs.k8s.io/)
# - kubectl
# - pip install konduktor-nightly
#
# Environment variables:
# - PLUTO_API_TOKEN: Required for authentication
# - PLUTO_RUN_ID: Optional, will be generated if not set
# - KEEP_CLUSTER: Set to "true" to keep the kind cluster after test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CLUSTER_NAME="pluto-multinode-test"
IMAGE_NAME="pluto-multinode-test:latest"
JOBSET_VERSION="v0.8.0"
KUEUE_VERSION="v0.10.1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        error "docker is not installed"
    fi

    if ! command -v kind &> /dev/null; then
        error "kind is not installed. Install from https://kind.sigs.k8s.io/"
    fi

    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed"
    fi

    if [[ -z "${PLUTO_API_TOKEN:-}" ]]; then
        error "PLUTO_API_TOKEN environment variable is required"
    fi

    log "Prerequisites OK"
}

# Install konduktor CLI
install_konduktor() {
    log "Installing konduktor CLI..."

    if command -v konduktor &> /dev/null; then
        log "konduktor already installed: $(konduktor --version 2>/dev/null || echo 'version unknown')"
        return
    fi

    pip install konduktor-nightly

    log "konduktor installed"
}

# Create kind cluster
create_cluster() {
    log "Creating kind cluster: ${CLUSTER_NAME}"

    # Delete existing cluster if it exists
    if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
        warn "Cluster ${CLUSTER_NAME} already exists, deleting..."
        kind delete cluster --name "${CLUSTER_NAME}"
    fi

    kind create cluster --name "${CLUSTER_NAME}" --wait 60s
    log "Cluster created"
}

# Install JobSet controller (required by konduktor)
install_jobset() {
    log "Installing JobSet controller ${JOBSET_VERSION}..."

    kubectl apply --server-side -f "https://github.com/kubernetes-sigs/jobset/releases/download/${JOBSET_VERSION}/manifests.yaml"

    log "Waiting for JobSet controller to be ready..."
    kubectl wait --for=condition=available --timeout=120s deployment/jobset-controller-manager -n jobset-system

    log "JobSet controller installed"
}

# Install Kueue controller
install_kueue() {
    log "Installing Kueue ${KUEUE_VERSION}..."

    kubectl apply --server-side -f "https://github.com/kubernetes-sigs/kueue/releases/download/${KUEUE_VERSION}/manifests.yaml"

    log "Waiting for Kueue controller to be ready..."
    kubectl wait --for=condition=available --timeout=120s deployment/kueue-controller-manager -n kueue-system

    log "Kueue installed"
}

# Create ClusterQueue and LocalQueue for konduktor
create_queues() {
    log "Creating ResourceFlavor, ClusterQueue, and LocalQueue..."

    # Apply the kueue configuration from file
    kubectl apply -f "${SCRIPT_DIR}/kueue-config.yaml"

    # Wait for queues to be ready
    log "Waiting for ClusterQueue to be ready..."
    sleep 5  # Give kueue time to process

    # Verify queues are created
    kubectl get clusterqueue cluster-queue
    kubectl get localqueue user-queue -n default

    log "Queues created"
}

# Build and load test image
build_image() {
    log "Building test image: ${IMAGE_NAME}"

    docker build -t "${IMAGE_NAME}" -f "${SCRIPT_DIR}/Dockerfile" "${REPO_ROOT}"

    log "Loading image into kind cluster..."
    kind load docker-image "${IMAGE_NAME}" --name "${CLUSTER_NAME}"

    log "Image loaded"
}

# Run the multinode test using konduktor CLI
run_test() {
    local run_id="${PLUTO_RUN_ID:-k8s-multinode-$(date +%s)-$(head /dev/urandom | tr -dc a-z0-9 | head -c 8)}"

    log "Running multinode test with konduktor"
    log "PLUTO_RUN_ID: ${run_id}"

    # Export environment variables for konduktor
    export PLUTO_RUN_ID="${run_id}"
    export PLUTO_API_TOKEN="${PLUTO_API_TOKEN}"
    export PLUTO_PROJECT="testing-ci"

    # Launch the job using konduktor CLI
    log "Launching job with: konduktor launch ${SCRIPT_DIR}/multinode-job.yaml"
    konduktor launch "${SCRIPT_DIR}/multinode-job.yaml" \
        --env "PLUTO_RUN_ID=${run_id}" \
        --env "PLUTO_API_TOKEN=${PLUTO_API_TOKEN}" \
        --env "PLUTO_PROJECT=testing-ci"

    # Wait for the job to complete
    log "Waiting for job to complete (timeout: 5m)..."
    local start_time
    start_time=$(date +%s)
    local timeout=300  # 5 minutes

    while true; do
        local current_time
        current_time=$(date +%s)
        local elapsed=$((current_time - start_time))

        if [[ ${elapsed} -gt ${timeout} ]]; then
            warn "Timeout waiting for job. Fetching logs..."
            kubectl logs -l jobset.sigs.k8s.io/jobset-name=pluto-multinode-e2e-test --all-containers=true || true
            error "Timeout waiting for job to complete"
        fi

        # Check job status via konduktor or kubectl
        # konduktor queue shows job status
        if konduktor queue 2>/dev/null | grep -q "pluto-multinode-e2e-test.*SUCCEEDED"; then
            log "Job completed successfully!"
            break
        fi

        if konduktor queue 2>/dev/null | grep -q "pluto-multinode-e2e-test.*FAILED"; then
            warn "Job failed. Fetching logs..."
            konduktor logs pluto-multinode-e2e-test || true
            kubectl logs -l jobset.sigs.k8s.io/jobset-name=pluto-multinode-e2e-test --all-containers=true || true
            error "Job failed"
        fi

        # Also check via kubectl for JobSet completion
        local jobset_status
        jobset_status=$(kubectl get jobset pluto-multinode-e2e-test -o jsonpath='{.status.conditions[?(@.type=="Completed")].status}' 2>/dev/null || echo "")
        if [[ "${jobset_status}" == "True" ]]; then
            log "JobSet completed successfully!"
            break
        fi

        log "Waiting... (${elapsed}s elapsed)"
        sleep 5
    done

    # Get logs from the job
    log "Fetching job logs..."
    konduktor logs pluto-multinode-e2e-test 2>/dev/null || \
        kubectl logs -l jobset.sigs.k8s.io/jobset-name=pluto-multinode-e2e-test --all-containers=true || true

    log "Test completed successfully!"
    log "Run ID: ${run_id}"
    log "View run at: https://pluto.trainy.ai/testing-ci?run=${run_id}"
}

# Cleanup
cleanup() {
    log "Cleaning up..."

    # Cancel the job if it's still running
    konduktor cancel pluto-multinode-e2e-test 2>/dev/null || true

    # Optionally delete the cluster (controlled by KEEP_CLUSTER env var)
    if [[ "${KEEP_CLUSTER:-false}" != "true" ]]; then
        log "Deleting kind cluster..."
        kind delete cluster --name "${CLUSTER_NAME}" || true
    else
        warn "Keeping cluster (KEEP_CLUSTER=true). Delete manually with: kind delete cluster --name ${CLUSTER_NAME}"
    fi

    log "Cleanup complete"
}

# Main
main() {
    log "Starting k8s multinode E2E test with konduktor"
    log ""
    log "This test demonstrates:"
    log "  1. Launching a 2-node distributed job with konduktor"
    log "  2. Using torchrun for distributed coordination"
    log "  3. All nodes sharing the same PLUTO_RUN_ID"
    log "  4. Each node logging metrics to the same Pluto run"
    log ""

    # Set up trap for cleanup on exit
    trap cleanup EXIT

    check_prerequisites
    install_konduktor
    create_cluster
    install_jobset
    install_kueue
    create_queues
    build_image
    run_test

    log "All tests passed!"
}

main "$@"
