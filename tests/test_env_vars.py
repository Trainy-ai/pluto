import logging
import os

from mlop.sets import setup


class TestMLOPDebugLevel:
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
            os.environ['MLOP_DEBUG_LEVEL'] = env_val
            settings = setup()
            assert settings.x_log_level == expected_level
            del os.environ['MLOP_DEBUG_LEVEL']

    def test_case_insensitive(self):
        """Test lowercase and mixed case"""
        os.environ['MLOP_DEBUG_LEVEL'] = 'debug'
        settings = setup()
        assert settings.x_log_level == 10
        del os.environ['MLOP_DEBUG_LEVEL']

        os.environ['MLOP_DEBUG_LEVEL'] = 'Info'
        settings = setup()
        assert settings.x_log_level == 20
        del os.environ['MLOP_DEBUG_LEVEL']

    def test_numeric_values(self):
        """Test numeric strings"""
        os.environ['MLOP_DEBUG_LEVEL'] = '15'
        settings = setup()
        assert settings.x_log_level == 15
        del os.environ['MLOP_DEBUG_LEVEL']

        os.environ['MLOP_DEBUG_LEVEL'] = '25'
        settings = setup()
        assert settings.x_log_level == 25
        del os.environ['MLOP_DEBUG_LEVEL']

    def test_precedence(self):
        """Test that function params override env var"""
        os.environ['MLOP_DEBUG_LEVEL'] = 'DEBUG'
        settings = setup({'x_log_level': 30})
        assert settings.x_log_level == 30  # Dict wins
        del os.environ['MLOP_DEBUG_LEVEL']

    def test_default_when_not_set(self):
        """Test default value when env var not set"""
        # Make sure env var is not set
        if 'MLOP_DEBUG_LEVEL' in os.environ:
            del os.environ['MLOP_DEBUG_LEVEL']

        settings = setup()
        assert settings.x_log_level == 16  # Default value

    def test_invalid_value_warning(self, caplog):
        """Test warning on invalid value"""
        os.environ['MLOP_DEBUG_LEVEL'] = 'INVALID'
        with caplog.at_level(logging.WARNING):
            settings = setup()
            assert 'invalid MLOP_DEBUG_LEVEL' in caplog.text
            assert settings.x_log_level == 16  # Falls back to default
        del os.environ['MLOP_DEBUG_LEVEL']

    def test_empty_string(self):
        """Test empty string uses default"""
        os.environ['MLOP_DEBUG_LEVEL'] = ''
        settings = setup()
        # Empty string is falsy, so env var will be None and default is used
        assert settings.x_log_level == 16
        del os.environ['MLOP_DEBUG_LEVEL']
