#!/bin/bash
# K8s multinode E2E test using konduktor CLI
#
# Tests that multiple nodes can log to the same Pluto run via shared PLUTO_RUN_ID.
#
# Prerequisites:
# - docker, kind, kubectl
# - PLUTO_API_TOKEN environment variable
#
# Options:
# - KEEP_CLUSTER=true to keep cluster after test

set -euox pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CLUSTER_NAME="pluto-multinode-test"
IMAGE_NAME="pluto-multinode-test:local"
JOB_NAME="pluto-multinode-e2e-test"

log() { echo -e "\033[0;32m[INFO]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error() { echo -e "\033[0;31m[ERROR]\033[0m $*"; exit 1; }

check_prerequisites() {
    log "Checking prerequisites..."
    command -v docker &>/dev/null || error "docker not installed"
    command -v kind &>/dev/null || error "kind not installed"
    command -v kubectl &>/dev/null || error "kubectl not installed"
    [[ -n "${PLUTO_API_TOKEN:-}" ]] || error "PLUTO_API_TOKEN not set"
    log "Prerequisites OK"
}

install_konduktor() {
    log "Installing konduktor CLI..."
    if ! command -v konduktor &>/dev/null; then
        pip install konduktor-nightly
    fi
    log "konduktor ready"
}

create_cluster() {
    log "Creating kind cluster..."
    if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
        kind delete cluster --name "${CLUSTER_NAME}"
    fi
    kind create cluster --name "${CLUSTER_NAME}" --wait 60s
    log "Cluster created"
}

setup_cluster() {
    log "Setting up cluster components..."

    # Install required controllers (using kubectl for cluster setup only)
    kubectl apply --server-side -f "https://github.com/kubernetes-sigs/jobset/releases/download/v0.8.0/manifests.yaml" >/dev/null
    kubectl apply --server-side -f "https://github.com/kubernetes-sigs/kueue/releases/download/v0.10.1/manifests.yaml" >/dev/null

    kubectl wait --for=condition=available --timeout=120s deployment/jobset-controller-manager -n jobset-system >/dev/null
    kubectl wait --for=condition=available --timeout=120s deployment/kueue-controller-manager -n kueue-system >/dev/null

    # Apply queue configuration with retry (webhook may take time to initialize after deployment is available)
    local retries=5
    for ((i=1; i<=retries; i++)); do
        if kubectl apply -f "${SCRIPT_DIR}/kueue-config.yaml" >/dev/null 2>&1; then
            break
        fi
        if [[ $i -eq $retries ]]; then
            error "Failed to apply kueue-config.yaml after ${retries} retries"
        fi
        log "Waiting for kueue webhook (attempt ${i}/${retries})..."
        sleep 5
    done
    sleep 5

    log "Cluster setup complete"
}

build_image() {
    log "Building test image..."
    docker build -t "${IMAGE_NAME}" -f "${SCRIPT_DIR}/Dockerfile" "${REPO_ROOT}" -q
    kind load docker-image "${IMAGE_NAME}" --name "${CLUSTER_NAME}" >/dev/null
    log "Image ready"
}

run_test() {
    local run_id="${PLUTO_RUN_ID:-k8s-multinode-$(date +%s)-$(head /dev/urandom | tr -dc a-z0-9 | head -c 8)}"
    log "Running test with PLUTO_RUN_ID: ${run_id}"

    # Launch job using konduktor CLI (following smoke test pattern)
    log "Launching job with konduktor..."
    konduktor launch --name "${JOB_NAME}" -y "${SCRIPT_DIR}/multinode-job.yaml" \
        --env "PLUTO_RUN_ID=${run_id}" \
        --env "PLUTO_API_TOKEN=${PLUTO_API_TOKEN}" \
        --env "PLUTO_PROJECT=testing-ci"

    log "Job launched. Checking initial status..."
    konduktor status || true
    kubectl get pods -A || true

    # Wait for job completion using konduktor status (following smoke test pattern)
    local timeout=300
    local start=$(date +%s)
    local last_status_log=0
    while true; do
        local elapsed=$(($(date +%s) - start))
        [[ ${elapsed} -gt ${timeout} ]] && {
            log "Timeout - fetching debug info..."
            konduktor status || true
            kubectl get pods -A || true
            kubectl describe pods -l jobset.sigs.k8s.io/jobset-name="${JOB_NAME}" 2>/dev/null || true
            konduktor logs --no-follow "${JOB_NAME}" || true
            error "Timeout waiting for job after ${timeout}s"
        }

        # Log status every 30 seconds
        if [[ $((elapsed - last_status_log)) -ge 30 ]]; then
            log "Still waiting (${elapsed}s elapsed)..."
            konduktor status 2>/dev/null || true
            kubectl get pods -A 2>/dev/null | grep -E "NAME|${JOB_NAME}" || true
            last_status_log=${elapsed}
        fi

        # Check job status using konduktor status | grep pattern
        local status_output
        status_output=$(konduktor status 2>/dev/null || true)
        if echo "${status_output}" | grep -q "${JOB_NAME}.*COMPLETED"; then
            log "Job completed successfully"
            break
        fi

        if echo "${status_output}" | grep -q "${JOB_NAME}.*FAILED"; then
            log "Job failed - fetching logs..."
            kubectl get pods -A || true
            kubectl describe pods -l jobset.sigs.k8s.io/jobset-name="${JOB_NAME}" 2>/dev/null || true
            konduktor logs --no-follow "${JOB_NAME}" || true
            error "Job failed"
        fi

        sleep 5
    done

    # Fetch and display job logs
    log "Fetching job logs..."
    local logs
    logs=$(konduktor logs --no-follow "${JOB_NAME}" 2>&1 || true)
    echo "${logs}"

    # Extract and prominently display the Pluto experiment URL
    local pluto_url
    pluto_url=$(echo "${logs}" | grep -oP 'PLUTO_EXPERIMENT_URL=\K[^\s]+' | head -1 || true)

    echo ""
    echo "========================================"
    echo "         TEST RESULTS"
    echo "========================================"
    echo "Run ID:     ${run_id}"
    if [[ -n "${pluto_url}" ]]; then
        echo "Pluto URL:  ${pluto_url}"
        echo ""
        echo "View your experiment at:"
        echo "  ${pluto_url}"
    else
        warn "Could not extract Pluto URL from logs"
    fi
    echo "========================================"
    echo ""

    log "Test passed!"
}

cleanup() {
    log "Cleaning up..."
    konduktor down "${JOB_NAME}" 2>/dev/null || true
    if [[ "${KEEP_CLUSTER:-false}" != "true" ]]; then
        kind delete cluster --name "${CLUSTER_NAME}" 2>/dev/null || true
    fi
}

main() {
    trap cleanup EXIT
    check_prerequisites
    install_konduktor
    create_cluster
    setup_cluster
    build_image
    run_test
    log "All tests passed!"
}

main "$@"
