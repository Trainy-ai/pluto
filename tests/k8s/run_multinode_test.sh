#!/bin/bash
# K8s multinode E2E test runner
# This script sets up a kind cluster, installs JobSet, and runs the multinode test.
#
# Prerequisites:
# - docker
# - kind (https://kind.sigs.k8s.io/)
# - kubectl
#
# Environment variables:
# - PLUTO_API_TOKEN: Required for authentication
# - PLUTO_RUN_ID: Optional, will be generated if not set

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CLUSTER_NAME="pluto-multinode-test"
IMAGE_NAME="pluto-multinode-test:latest"
JOBSET_VERSION="v0.8.0"

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

# Install JobSet controller
install_jobset() {
    log "Installing JobSet controller ${JOBSET_VERSION}..."

    kubectl apply --server-side -f "https://github.com/kubernetes-sigs/jobset/releases/download/${JOBSET_VERSION}/manifests.yaml"

    log "Waiting for JobSet controller to be ready..."
    kubectl wait --for=condition=available --timeout=120s deployment/jobset-controller-manager -n jobset-system

    log "JobSet controller installed"
}

# Build and load test image
build_image() {
    log "Building test image: ${IMAGE_NAME}"

    docker build -t "${IMAGE_NAME}" -f "${SCRIPT_DIR}/Dockerfile" "${REPO_ROOT}"

    log "Loading image into kind cluster..."
    kind load docker-image "${IMAGE_NAME}" --name "${CLUSTER_NAME}"

    log "Image loaded"
}

# Run the multinode test
run_test() {
    local run_id="${PLUTO_RUN_ID:-k8s-multinode-$(date +%s)-$(head /dev/urandom | tr -dc a-z0-9 | head -c 8)}"
    local manifest_file="${SCRIPT_DIR}/jobset-multinode-test.yaml"

    log "Running multinode test with PLUTO_RUN_ID: ${run_id}"

    # Create a temporary manifest with the actual values
    local tmp_manifest
    tmp_manifest=$(mktemp)
    sed -e "s/PLACEHOLDER_RUN_ID/${run_id}/g" \
        -e "s/PLACEHOLDER_TOKEN/${PLUTO_API_TOKEN}/g" \
        "${manifest_file}" > "${tmp_manifest}"

    # Apply the JobSet
    log "Applying JobSet manifest..."
    kubectl apply -f "${tmp_manifest}"

    # Wait for the JobSet to complete
    log "Waiting for JobSet to complete (timeout: 5m)..."
    local start_time
    start_time=$(date +%s)
    local timeout=300  # 5 minutes

    while true; do
        local current_time
        current_time=$(date +%s)
        local elapsed=$((current_time - start_time))

        if [[ ${elapsed} -gt ${timeout} ]]; then
            error "Timeout waiting for JobSet to complete"
        fi

        # Check JobSet status
        local status
        status=$(kubectl get jobset pluto-multinode-test -o jsonpath='{.status.conditions[?(@.type=="Completed")].status}' 2>/dev/null || echo "")

        if [[ "${status}" == "True" ]]; then
            log "JobSet completed successfully!"
            break
        fi

        # Check for failure
        local failed
        failed=$(kubectl get jobset pluto-multinode-test -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
        if [[ "${failed}" == "True" ]]; then
            warn "JobSet failed. Fetching logs..."
            kubectl logs -l jobset.sigs.k8s.io/jobset-name=pluto-multinode-test --all-containers=true || true
            error "JobSet failed"
        fi

        log "Waiting... (${elapsed}s elapsed)"
        sleep 5
    done

    # Get logs from all pods
    log "Fetching pod logs..."
    kubectl logs -l jobset.sigs.k8s.io/jobset-name=pluto-multinode-test --all-containers=true

    # Cleanup
    rm -f "${tmp_manifest}"

    log "Test completed successfully!"
    log "Run ID: ${run_id}"
    log "View run at: https://pluto.trainy.ai/testing-ci?run=${run_id}"
}

# Cleanup
cleanup() {
    log "Cleaning up..."

    # Delete JobSet if it exists
    kubectl delete jobset pluto-multinode-test --ignore-not-found=true || true

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
    log "Starting k8s multinode E2E test"

    # Set up trap for cleanup on exit
    trap cleanup EXIT

    check_prerequisites
    create_cluster
    install_jobset
    build_image
    run_test

    log "All tests passed!"
}

main "$@"
