"""Tests for log sanitization of secrets before backend upload."""

import pytest


# === AWS ===


class TestAWSKeys:
    def test_aws_access_key_id(self):
        line = "Using AWS key AKIAIOSFODNN7EXAMPLE for auth"
        assert "AKIAIOSFODNN7EXAMPLE" not in sanitize(line)

    def test_aws_secret_access_key_in_assignment(self):
        line = "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        assert "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" not in sanitize(line)

    def test_aws_secret_in_print(self):
        line = "AWS key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        assert "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" not in sanitize(line)


# === Connection strings ===


class TestConnectionStrings:
    def test_postgres_connection_string(self):
        line = "Config: {'db_url': 'postgresql://admin:supersecretpassword@db.example.com:5432/mldata'}"
        assert "supersecretpassword" not in sanitize(line)

    def test_mysql_connection_string(self):
        line = "mysql://root:hunter2@localhost:3306/mydb"
        assert "hunter2" not in sanitize(line)

    def test_redis_connection_string(self):
        line = "redis://default:mypassword@redis.example.com:6379/0"
        assert "mypassword" not in sanitize(line)

    def test_mongodb_connection_string(self):
        line = "mongodb://admin:p%40ssw0rd@mongo.example.com:27017/db"
        assert "p%40ssw0rd" not in sanitize(line)

    def test_connection_string_preserves_structure(self):
        """The host/port/db should still be visible after redaction."""
        line = "postgresql://admin:secret@db.example.com:5432/mldata"
        result = sanitize(line)
        assert "db.example.com" in result
        assert "secret" not in result


# === API tokens with known prefixes ===


class TestAPITokens:
    def test_huggingface_token(self):
        line = "token=hf_abc123DEF456ghi789JKL012mno345PQR678"
        assert "hf_abc123DEF456ghi789JKL012mno345PQR678" not in sanitize(line)

    def test_huggingface_token_in_url(self):
        line = "https://huggingface.co/model/resolve/main/model.bin?token=hf_abcDEF123456"
        assert "hf_abcDEF123456" not in sanitize(line)

    def test_github_pat(self):
        line = "GITHUB_TOKEN=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef012345"
        assert "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef012345" not in sanitize(line)

    def test_github_oauth(self):
        line = "token: gho_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef012345"
        assert "gho_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef012345" not in sanitize(line)

    def test_gitlab_token(self):
        line = "GITLAB_TOKEN=glpat-xxxxxxxxxxxxxxxxxxxx"
        assert "glpat-xxxxxxxxxxxxxxxxxxxx" not in sanitize(line)

    def test_openai_key(self):
        line = "api_key=sk-proj-abcdefghijklmnopqrstuvwxyz123456"
        assert "sk-proj-abcdefghijklmnopqrstuvwxyz123456" not in sanitize(line)

    def test_openai_legacy_key(self):
        line = "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz12345678901234567890"
        assert "sk-abcdefghijklmnopqrstuvwxyz12345678901234567890" not in sanitize(line)

    def test_slack_bot_token(self):
        # Build at runtime to avoid GitHub push protection false positives
        prefix = "xoxb"
        line = f"SLACK_TOKEN={prefix}-000000000000-0000000000000-FAKEFAKEFAKEFAKE"
        assert prefix + "-" not in sanitize(line)

    def test_slack_user_token(self):
        prefix = "xoxp"
        line = f"token={prefix}-000000000000-0000000000000-FAKEFAKEFAKEFAKE"
        assert prefix + "-" not in sanitize(line)

    def test_stripe_secret_key(self):
        # Build at runtime to avoid GitHub push protection false positives
        prefix = "sk" + "_live_"
        line = f"{prefix}FAKEFAKEFAKEFAKEFAKEFAKE00000000"
        assert prefix not in sanitize(line)

    def test_stripe_test_key(self):
        prefix = "sk" + "_test_"
        line = f"{prefix}FAKEFAKEFAKEFAKEFAKEFAKE00000000"
        assert prefix not in sanitize(line)

    def test_pluto_api_token(self):
        line = "Using token mlpi_gcXcamO2fKt4MLwD"
        assert "mlpi_gcXcamO2fKt4MLwD" not in sanitize(line)

    def test_pypi_token(self):
        line = "TWINE_PASSWORD=pypi-AgEIcHlwaS5vcmcCJGNmNTI5MjA3LWQxYjAtNDRlYy0"
        assert "pypi-" not in sanitize(line)

    def test_npm_token(self):
        line = "//registry.npmjs.org/:_authToken=npm_abcdefghijklmnopqrstuvwxyz123456"
        assert "npm_abcdefghijklmnopqrstuvwxyz123456" not in sanitize(line)


# === Bearer / Authorization headers ===


class TestAuthHeaders:
    def test_bearer_token(self):
        line = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in sanitize(line)

    def test_basic_auth_header(self):
        line = "Authorization: Basic dXNlcm5hbWU6cGFzc3dvcmQ="
        assert "dXNlcm5hbWU6cGFzc3dvcmQ=" not in sanitize(line)


# === Private keys ===


class TestPrivateKeys:
    def test_rsa_private_key_header(self):
        line = "-----BEGIN RSA PRIVATE KEY-----"
        assert "PRIVATE KEY" not in sanitize(line)

    def test_generic_private_key_header(self):
        line = "-----BEGIN PRIVATE KEY-----"
        assert "PRIVATE KEY" not in sanitize(line)

    def test_ec_private_key_header(self):
        line = "-----BEGIN EC PRIVATE KEY-----"
        assert "PRIVATE KEY" not in sanitize(line)


# === Key=value patterns with secret-like names ===


class TestKeyValueAssignments:
    def test_password_equals(self):
        line = "password=hunter2"
        assert "hunter2" not in sanitize(line)

    def test_password_colon(self):
        line = "password: hunter2"
        assert "hunter2" not in sanitize(line)

    def test_api_key_equals(self):
        line = 'api_key="some-secret-value-12345"'
        assert "some-secret-value-12345" not in sanitize(line)

    def test_secret_key_equals(self):
        line = "secret_key=abcdef1234567890"
        assert "abcdef1234567890" not in sanitize(line)

    def test_access_token_equals(self):
        line = "access_token=eyJhbGciOiJIUzI1NiJ9.eyJ0ZXN0IjoiMTIzIn0.abc123"
        assert "eyJhbGciOiJIUzI1NiJ9" not in sanitize(line)

    def test_db_password_in_dict(self):
        line = "{'db_password': 'supersecret123'}"
        assert "supersecret123" not in sanitize(line)

    def test_secret_in_json(self):
        line = '"client_secret": "my-app-secret-value"'
        assert "my-app-secret-value" not in sanitize(line)


# === Things that should NOT be redacted ===


class TestFalsePositives:
    def test_normal_training_log(self):
        line = "Epoch 5/10: loss=0.0032, accuracy=0.9871"
        assert sanitize(line) == line

    def test_normal_metric_log(self):
        line = "val/loss: 0.25, train/loss: 0.12"
        assert sanitize(line) == line

    def test_model_architecture_log(self):
        line = "Linear(in_features=768, out_features=512, bias=True)"
        assert sanitize(line) == line

    def test_file_path(self):
        line = "Saving checkpoint to /home/user/models/checkpoint-1000"
        assert sanitize(line) == line

    def test_gpu_info(self):
        line = "GPU 0: NVIDIA A100 80GB, memory: 81920MB, utilization: 95%"
        assert sanitize(line) == line

    def test_short_password_value_not_over_triggered(self):
        """Very short values after 'password=' might be acceptable to flag,
        but normal words like 'True' or 'None' should not be redacted elsewhere."""
        line = "use_password=True"
        # This is ambiguous — it's fine either way. The key thing is
        # the rest of the line isn't mangled.
        result = sanitize(line)
        assert "True" in result or "REDACTED" in result

    def test_learning_rate_not_redacted(self):
        line = "lr=0.001, batch_size=32, epochs=10"
        assert sanitize(line) == line

    def test_timestamp_not_redacted(self):
        line = "2024-01-15 10:30:45.123 INFO Training started"
        assert sanitize(line) == line

    def test_sklearn_params_not_redacted(self):
        line = "RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)"
        assert sanitize(line) == line


# Placeholder — import will be filled in once we implement pluto/sanitize.py
def sanitize(line: str) -> str:
    """Placeholder that will be replaced with actual import."""
    from pluto.sanitize import SecretSanitizer

    _sanitizer = SecretSanitizer()
    return _sanitizer.sanitize(line)
