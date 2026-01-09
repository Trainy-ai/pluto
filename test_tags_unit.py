#!/usr/bin/env python3
"""Unit tests for tags API payload generation (no server connection needed)."""

import json
from mlop.sets import Settings
from mlop.api import make_compat_update_tags_v1


def test_api_payload():
    """Test that API payload has correct format for HTTP endpoint."""
    # Create a settings object
    settings = Settings()
    settings.mode = 'noop'
    settings._op_id = 12345  # Numeric ID
    settings.project = 'test-project'

    # Generate payload
    payload = make_compat_update_tags_v1(settings, ['tag1', 'tag2', 'tag3'])

    # Decode and verify
    payload_dict = json.loads(payload.decode())

    print("Generated payload:", payload_dict)

    # Verify structure matches HTTP endpoint expectations
    assert 'runId' in payload_dict, "Payload should have 'runId' field"
    assert 'tags' in payload_dict, "Payload should have 'tags' field"
    assert 'projectName' not in payload_dict, "Payload should NOT have 'projectName' field"

    # Verify values
    assert payload_dict['runId'] == 12345, f"runId should be numeric 12345, got {payload_dict['runId']}"
    assert isinstance(payload_dict['runId'], int), f"runId should be integer, got {type(payload_dict['runId'])}"
    assert payload_dict['tags'] == ['tag1', 'tag2', 'tag3'], f"tags should match input, got {payload_dict['tags']}"

    print("✅ All payload tests passed!")


def test_url_endpoint():
    """Test that URL points to HTTP endpoint, not tRPC."""
    settings = Settings()
    settings.mode = 'noop'
    settings.update_host()  # Initialize URLs

    print(f"Tags update URL: {settings.url_update_tags}")

    assert '/api/runs/tags/update' in settings.url_update_tags, \
        f"URL should be HTTP endpoint, got {settings.url_update_tags}"
    assert '/trpc/' not in settings.url_update_tags, \
        f"URL should not be tRPC endpoint, got {settings.url_update_tags}"

    print("✅ URL endpoint test passed!")


if __name__ == '__main__':
    print("Testing tags update API payload format...\n")
    test_api_payload()
    print()
    test_url_endpoint()
    print("\n✅ All tests passed!")
