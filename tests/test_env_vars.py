import logging
import os
import warnings

from pluto.sets import setup


class TestPLUTODebugLevel:
    def test_string_values(self):
        """Test DEBUG, INFO, WARNING, ERROR, CRITICAL"""
        test_cases = [
            ('DEBUG', 10),
            ('INFO', 20),
            ('WARNING', 30),
            ('ERROR', 40),
            ('CRITICAL', 50),
        ]
        for env_val, expected_level in test_cases:
            os.environ['PLUTO_DEBUG_LEVEL'] = env_val
            settings = setup()
            assert settings.x_log_level == expected_level
            del os.environ['PLUTO_DEBUG_LEVEL']

    def test_case_insensitive(self):
        """Test lowercase and mixed case"""
        os.environ['PLUTO_DEBUG_LEVEL'] = 'debug'
        settings = setup()
        assert settings.x_log_level == 10
        del os.environ['PLUTO_DEBUG_LEVEL']

        os.environ['PLUTO_DEBUG_LEVEL'] = 'Info'
        settings = setup()
        assert settings.x_log_level == 20
        del os.environ['PLUTO_DEBUG_LEVEL']

    def test_numeric_values(self):
        """Test numeric strings"""
        os.environ['PLUTO_DEBUG_LEVEL'] = '15'
        settings = setup()
        assert settings.x_log_level == 15
        del os.environ['PLUTO_DEBUG_LEVEL']

        os.environ['PLUTO_DEBUG_LEVEL'] = '25'
        settings = setup()
        assert settings.x_log_level == 25
        del os.environ['PLUTO_DEBUG_LEVEL']

    def test_precedence(self):
        """Test that function params override env var"""
        os.environ['PLUTO_DEBUG_LEVEL'] = 'DEBUG'
        settings = setup({'x_log_level': 30})
        assert settings.x_log_level == 30  # Dict wins
        del os.environ['PLUTO_DEBUG_LEVEL']

    def test_default_when_not_set(self):
        """Test default value when env var not set"""
        # Make sure env var is not set
        if 'PLUTO_DEBUG_LEVEL' in os.environ:
            del os.environ['PLUTO_DEBUG_LEVEL']

        settings = setup()
        assert settings.x_log_level == 16  # Default value

    def test_invalid_value_warning(self, caplog):
        """Test warning on invalid value"""
        os.environ['PLUTO_DEBUG_LEVEL'] = 'INVALID'
        with caplog.at_level(logging.WARNING):
            settings = setup()
            assert 'invalid PLUTO_DEBUG_LEVEL' in caplog.text
            assert settings.x_log_level == 16  # Falls back to default
        del os.environ['PLUTO_DEBUG_LEVEL']

    def test_empty_string(self):
        """Test empty string uses default"""
        os.environ['PLUTO_DEBUG_LEVEL'] = ''
        settings = setup()
        # Empty string is falsy, so env var will be None and default is used
        assert settings.x_log_level == 16
        del os.environ['PLUTO_DEBUG_LEVEL']


class TestPLUTOURLEnvironmentVariables:
    def test_url_app(self):
        """Test PLUTO_URL_APP environment variable"""
        os.environ['PLUTO_URL_APP'] = 'https://custom-app.example.com'
        settings = setup()
        assert settings.url_app == 'https://custom-app.example.com'
        del os.environ['PLUTO_URL_APP']

    def test_url_api(self):
        """Test PLUTO_URL_API environment variable"""
        os.environ['PLUTO_URL_API'] = 'https://custom-api.example.com'
        settings = setup()
        assert settings.url_api == 'https://custom-api.example.com'
        del os.environ['PLUTO_URL_API']

    def test_url_ingest(self):
        """Test PLUTO_URL_INGEST environment variable"""
        os.environ['PLUTO_URL_INGEST'] = 'https://custom-ingest.example.com'
        settings = setup()
        assert settings.url_ingest == 'https://custom-ingest.example.com'
        del os.environ['PLUTO_URL_INGEST']

    def test_url_py(self):
        """Test PLUTO_URL_PY environment variable"""
        os.environ['PLUTO_URL_PY'] = 'https://custom-py.example.com'
        settings = setup()
        assert settings.url_py == 'https://custom-py.example.com'
        del os.environ['PLUTO_URL_PY']

    def test_all_urls(self):
        """Test all URL environment variables together"""
        os.environ['PLUTO_URL_APP'] = 'https://app.example.com'
        os.environ['PLUTO_URL_API'] = 'https://api.example.com'
        os.environ['PLUTO_URL_INGEST'] = 'https://ingest.example.com'
        os.environ['PLUTO_URL_PY'] = 'https://py.example.com'
        settings = setup()
        assert settings.url_app == 'https://app.example.com'
        assert settings.url_api == 'https://api.example.com'
        assert settings.url_ingest == 'https://ingest.example.com'
        assert settings.url_py == 'https://py.example.com'
        del os.environ['PLUTO_URL_APP']
        del os.environ['PLUTO_URL_API']
        del os.environ['PLUTO_URL_INGEST']
        del os.environ['PLUTO_URL_PY']

    def test_url_precedence(self):
        """Test that function params override env vars"""
        os.environ['PLUTO_URL_APP'] = 'https://env-app.example.com'
        os.environ['PLUTO_URL_API'] = 'https://env-api.example.com'
        os.environ['PLUTO_URL_INGEST'] = 'https://env-ingest.example.com'
        os.environ['PLUTO_URL_PY'] = 'https://env-py.example.com'
        settings = setup(
            {
                'url_app': 'https://param-app.example.com',
                'url_api': 'https://param-api.example.com',
                'url_ingest': 'https://param-ingest.example.com',
                'url_py': 'https://param-py.example.com',
            }
        )
        assert settings.url_app == 'https://param-app.example.com'  # Dict wins
        assert settings.url_api == 'https://param-api.example.com'
        assert settings.url_ingest == 'https://param-ingest.example.com'
        assert settings.url_py == 'https://param-py.example.com'
        del os.environ['PLUTO_URL_APP']
        del os.environ['PLUTO_URL_API']
        del os.environ['PLUTO_URL_INGEST']
        del os.environ['PLUTO_URL_PY']

    def test_default_urls_when_not_set(self):
        """Test default URLs when env vars not set"""
        # Make sure env vars are not set
        url_vars = [
            'PLUTO_URL_APP',
            'PLUTO_URL_API',
            'PLUTO_URL_INGEST',
            'PLUTO_URL_PY',
        ]
        for var in url_vars:
            if var in os.environ:
                del os.environ[var]

        settings = setup()
        # These should be the production defaults
        assert settings.url_app == 'https://pluto.trainy.ai'
        assert settings.url_api == 'https://pluto-api.trainy.ai'
        assert settings.url_ingest == 'https://pluto-ingest.trainy.ai'
        assert settings.url_py == 'https://pluto-py.trainy.ai'


class TestDeprecatedMLOPEnvVars:
    """Test that old MLOP_* env vars still work with deprecation warning."""

    def test_deprecated_debug_level(self):
        """Test deprecated MLOP_DEBUG_LEVEL still works with warning."""
        os.environ['MLOP_DEBUG_LEVEL'] = 'DEBUG'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            settings = setup()
            assert settings.x_log_level == 10
            # Check for deprecation warning
            assert any('MLOP_DEBUG_LEVEL' in str(warning.message) for warning in w)
        del os.environ['MLOP_DEBUG_LEVEL']

    def test_deprecated_url_app(self):
        """Test deprecated MLOP_URL_APP still works with warning."""
        os.environ['MLOP_URL_APP'] = 'https://old-app.example.com'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            settings = setup()
            assert settings.url_app == 'https://old-app.example.com'
            assert any('MLOP_URL_APP' in str(warning.message) for warning in w)
        del os.environ['MLOP_URL_APP']

    def test_new_env_var_takes_precedence_over_deprecated(self):
        """Test that PLUTO_* takes precedence over MLOP_* when both are set."""
        os.environ['MLOP_DEBUG_LEVEL'] = 'ERROR'
        os.environ['PLUTO_DEBUG_LEVEL'] = 'DEBUG'
        settings = setup()
        assert settings.x_log_level == 10  # DEBUG wins
        del os.environ['MLOP_DEBUG_LEVEL']
        del os.environ['PLUTO_DEBUG_LEVEL']
