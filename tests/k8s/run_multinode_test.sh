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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CLUSTER_NAME="pluto-multinode-test"
IMAGE_NAME="pluto-multinode-test:latest"

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

    # Install required controllers
    kubectl apply --server-side -f "https://github.com/kubernetes-sigs/jobset/releases/download/v0.8.0/manifests.yaml"
    kubectl apply --server-side -f "https://github.com/kubernetes-sigs/kueue/releases/download/v0.10.1/manifests.yaml"

    kubectl wait --for=condition=available --timeout=120s deployment/jobset-controller-manager -n jobset-system
    kubectl wait --for=condition=available --timeout=120s deployment/kueue-controller-manager -n kueue-system

    # Apply queue configuration
    kubectl apply -f "${SCRIPT_DIR}/kueue-config.yaml"
    sleep 5

    log "Cluster setup complete"
}

build_image() {
    log "Building test image..."
    docker build -t "${IMAGE_NAME}" -f "${SCRIPT_DIR}/Dockerfile" "${REPO_ROOT}"
    kind load docker-image "${IMAGE_NAME}" --name "${CLUSTER_NAME}"
    log "Image ready"
}

run_test() {
    local run_id="${PLUTO_RUN_ID:-k8s-multinode-$(date +%s)-$(head /dev/urandom | tr -dc a-z0-9 | head -c 8)}"
    log "Running test with PLUTO_RUN_ID: ${run_id}"

    konduktor launch "${SCRIPT_DIR}/multinode-job.yaml" \
        --env "PLUTO_RUN_ID=${run_id}" \
        --env "PLUTO_API_TOKEN=${PLUTO_API_TOKEN}" \
        --env "PLUTO_PROJECT=testing-ci"

    # Wait for completion
    local timeout=300
    local start=$(date +%s)
    while true; do
        local elapsed=$(($(date +%s) - start))
        [[ ${elapsed} -gt ${timeout} ]] && error "Timeout"

        if kubectl get jobset pluto-multinode-e2e-test -o jsonpath='{.status.conditions[?(@.type=="Completed")].status}' 2>/dev/null | grep -q "True"; then
            break
        fi
        if kubectl get jobset pluto-multinode-e2e-test -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null | grep -q "True"; then
            kubectl logs -l jobset.sigs.k8s.io/jobset-name=pluto-multinode-e2e-test --all-containers=true || true
            error "Job failed"
        fi
        sleep 5
    done

    kubectl logs -l jobset.sigs.k8s.io/jobset-name=pluto-multinode-e2e-test --all-containers=true || true
    log "Test passed! Run ID: ${run_id}"
}

cleanup() {
    log "Cleaning up..."
    konduktor cancel pluto-multinode-e2e-test 2>/dev/null || true
    if [[ "${KEEP_CLUSTER:-false}" != "true" ]]; then
        kind delete cluster --name "${CLUSTER_NAME}" || true
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
