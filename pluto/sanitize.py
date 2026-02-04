"""Sanitize secrets from log lines before uploading to the backend."""

import re
from typing import List, Tuple

REDACTED = '***REDACTED***'


class SecretSanitizer:
    """Detects and redacts secrets from log lines using compiled regex patterns."""

    def __init__(self) -> None:
        self._patterns: List[Tuple[re.Pattern, str]] = []
        self._build_patterns()

    def _build_patterns(self) -> None:
        # --- AWS ---
        # AWS access key IDs (always start with AKIA)
        self._patterns.append(
            (
                re.compile(r'AKIA[0-9A-Z]{16}'),
                REDACTED,
            )
        )
        # AWS secret access keys (40-char base64-ish, typically after an assignment)
        self._patterns.append(
            (
                re.compile(r'(?<=[=:\s])(?:[A-Za-z0-9/+=]{40})(?=\s|$|[\'",}])'),
                REDACTED,
            )
        )

        # --- Private keys ---
        self._patterns.append(
            (
                re.compile(r'-----BEGIN\s[\w\s]*PRIVATE KEY-----'),
                REDACTED,
            )
        )

        # --- Connection strings with embedded passwords ---
        # e.g. postgresql://user:password@host:port/db
        self._patterns.append(
            (
                re.compile(r'(://[^:/?#\s]+:)(\S+)(@)'),
                rf'\1{REDACTED}\3',
            )
        )

        # --- Authorization headers ---
        self._patterns.append(
            (
                re.compile(r'(Bearer\s+)\S+'),
                rf'\1{REDACTED}',
            )
        )
        self._patterns.append(
            (
                re.compile(r'(Basic\s+)\S+'),
                rf'\1{REDACTED}',
            )
        )

        # --- Known token prefixes ---
        known_prefixes = [
            r'sk-[A-Za-z0-9_-]{20,}',  # OpenAI / generic sk-
            r'sk_live_[A-Za-z0-9]{20,}',  # Stripe live
            r'sk_test_[A-Za-z0-9]{20,}',  # Stripe test
            r'hf_[A-Za-z0-9]{10,}',  # Hugging Face
            r'ghp_[A-Za-z0-9]{30,}',  # GitHub PAT
            r'gho_[A-Za-z0-9]{30,}',  # GitHub OAuth
            r'glpat-[A-Za-z0-9_-]{20,}',  # GitLab PAT
            r'xoxb-[A-Za-z0-9-]{20,}',  # Slack bot
            r'xoxp-[A-Za-z0-9-]{20,}',  # Slack user
            r'mlpi_[A-Za-z0-9]{10,}',  # Pluto API token
            r'pypi-[A-Za-z0-9]{10,}',  # PyPI
            r'npm_[A-Za-z0-9]{20,}',  # npm
        ]
        for prefix_pattern in known_prefixes:
            self._patterns.append(
                (
                    re.compile(prefix_pattern),
                    REDACTED,
                )
            )

        # --- Key=value assignments with secret-like variable names ---
        # Matches: password=val, secret_key="val", api_key='val', token: val, etc.
        secret_names = (
            r'password|passwd|secret|secret_key|api_key|apikey|'
            r'access_token|auth_token|token|'
            r'db_password|client_secret'
        )
        # key=value (unquoted)
        self._patterns.append(
            (
                re.compile(
                    rf"""(?i)(?:['"]?(?:{secret_names})['"]?)"""
                    rf"""(?:\s*[=:]\s*)"""
                    rf"""(?P<val>[^\s'",}}]+)"""
                ),
                rf'{REDACTED}',
            )
        )
        # key="value" or key='value'
        self._patterns.append(
            (
                re.compile(
                    rf"""(?i)(?:['"]?(?:{secret_names})['"]?)"""
                    rf"""(?:\s*[=:]\s*)"""
                    rf"""(?P<q>['"])(?P<val>.+?)(?P=q)"""
                ),
                rf'{REDACTED}',
            )
        )

    def sanitize(self, line: str) -> str:
        """Run all patterns against a line and return the sanitized version."""
        for pattern, replacement in self._patterns:
            line = pattern.sub(replacement, line)
        return line
