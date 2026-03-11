"""Tests for run resume support via display ID and numeric run ID.

This module tests the run resume feature where users can resume completed runs
by passing a display ID (like 'T0-123') or numeric run ID to pluto.init().
"""

import json
import os
import warnings

from pluto.api import make_compat_resume_v1
from pluto.sets import Settings, _classify_run_id, _is_display_id, setup


class TestIsDisplayId:
    """Test display ID pattern detection."""

    def test_standard_display_id(self):
        assert _is_display_id('T0-123') is True

    def test_short_prefix(self):
        assert _is_display_id('A-1') is True

    def test_long_prefix(self):
        assert _is_display_id('MMP-999') is True

    def test_multi_char_prefix(self):
        assert _is_display_id('PROJ-42') is True

    def test_plain_string(self):
        assert _is_display_id('my-external-id') is False

    def test_multiple_dashes(self):
        assert _is_display_id('my-run-123') is False

    def test_numeric_string(self):
        assert _is_display_id('12345') is False

    def test_no_number_after_dash(self):
        assert _is_display_id('T0-abc') is False

    def test_trailing_non_digit(self):
        assert _is_display_id('T0-123abc') is False

    def test_empty_string(self):
        assert _is_display_id('') is False

    def test_just_dash(self):
        assert _is_display_id('-123') is False

    def test_lowercase_prefix(self):
        assert _is_display_id('t0-123') is False

    def test_prefix_too_long(self):
        assert _is_display_id('ABCDE-1') is False

    def test_external_id_with_digit_suffix(self):
        assert _is_display_id('resume-12345678') is False


class TestClassifyRunId:
    """Test _classify_run_id routing logic."""

    def test_numeric_int(self):
        s = Settings()
        _classify_run_id(s, 42)
        assert s._resume_run_id == 42
        assert s._resume_display_id is None
        assert s._external_id is None

    def test_numeric_string(self):
        s = Settings()
        _classify_run_id(s, '12345')
        assert s._resume_run_id == 12345
        assert s._resume_display_id is None
        assert s._external_id is None

    def test_display_id(self):
        s = Settings()
        _classify_run_id(s, 'T0-123')
        assert s._resume_display_id == 'T0-123'
        assert s._resume_run_id is None
        assert s._external_id is None

    def test_external_id_string(self):
        s = Settings()
        _classify_run_id(s, 'my-ext-run-id')
        assert s._external_id == 'my-ext-run-id'
        assert s._resume_run_id is None
        assert s._resume_display_id is None

    def test_simple_string(self):
        s = Settings()
        _classify_run_id(s, 'training-run')
        assert s._external_id == 'training-run'

    def test_uuid_style(self):
        s = Settings()
        _classify_run_id(s, 'ddp-abc12345')
        assert s._external_id == 'ddp-abc12345'


class TestSetupEnvVarClassification:
    """Test that PLUTO_RUN_ID env var goes through _classify_run_id."""

    def test_env_var_display_id(self):
        os.environ['PLUTO_RUN_ID'] = 'T0-456'
        try:
            settings = setup()
            assert settings._resume_display_id == 'T0-456'
            assert settings._resume_run_id is None
            assert settings._external_id is None
        finally:
            del os.environ['PLUTO_RUN_ID']

    def test_env_var_numeric(self):
        os.environ['PLUTO_RUN_ID'] = '789'
        try:
            settings = setup()
            assert settings._resume_run_id == 789
            assert settings._resume_display_id is None
            assert settings._external_id is None
        finally:
            del os.environ['PLUTO_RUN_ID']

    def test_env_var_external_id(self):
        """Backward compat: plain string env var still sets _external_id."""
        os.environ['PLUTO_RUN_ID'] = 'my-ddp-run'
        try:
            settings = setup()
            assert settings._external_id == 'my-ddp-run'
            assert settings._resume_run_id is None
            assert settings._resume_display_id is None
        finally:
            del os.environ['PLUTO_RUN_ID']

    def test_deprecated_mlop_run_id_display_id(self):
        os.environ['MLOP_RUN_ID'] = 'MMP-10'
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                settings = setup()
                assert settings._resume_display_id == 'MMP-10'
                assert any('MLOP_RUN_ID' in str(warning.message) for warning in w)
        finally:
            del os.environ['MLOP_RUN_ID']


class TestMakeCompatResumeV1:
    """Test resume payload builder."""

    def test_resume_by_run_id(self):
        s = Settings()
        s._resume_run_id = 42
        payload = json.loads(make_compat_resume_v1(s))
        assert payload == {'runId': 42}

    def test_resume_by_display_id(self):
        s = Settings()
        s._resume_display_id = 'T0-123'
        payload = json.loads(make_compat_resume_v1(s))
        assert payload == {'displayId': 'T0-123'}

    def test_resume_by_external_id(self):
        s = Settings()
        s._external_id = 'my-ext-id'
        s.project = 'my-project'
        payload = json.loads(make_compat_resume_v1(s))
        assert payload == {'externalId': 'my-ext-id', 'projectName': 'my-project'}

    def test_resume_run_id_takes_priority(self):
        """_resume_run_id should take priority over display_id and external_id."""
        s = Settings()
        s._resume_run_id = 42
        s._resume_display_id = 'T0-123'
        s._external_id = 'ext-id'
        payload = json.loads(make_compat_resume_v1(s))
        assert payload == {'runId': 42}

    def test_empty_when_nothing_set(self):
        s = Settings()
        payload = json.loads(make_compat_resume_v1(s))
        assert payload == {}


class TestSettingsDefaultFields:
    """Test that new fields have correct defaults."""

    def test_resume_run_id_default(self):
        s = Settings()
        assert s._resume_run_id is None

    def test_resume_display_id_default(self):
        s = Settings()
        assert s._resume_display_id is None

    def test_url_resume_exists(self):
        s = Settings()
        s.update_host()
        assert hasattr(s, 'url_resume')
        assert 'api/runs/resume' in s.url_resume
