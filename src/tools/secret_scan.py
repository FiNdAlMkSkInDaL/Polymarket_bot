from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]

TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".env",
    ".txt",
    ".md",
    ".service",
    ".conf",
}

ALLOWED_ENV_FILENAMES = {
    ".env.example",
    ".env.sample",
    ".env.template",
    ".env.defaults",
}

FORBIDDEN_FILENAME_PATTERNS = (
    re.compile(r"(^|/)\.env(\.[^/]+)?$", re.IGNORECASE),
    re.compile(r"(^|/).+\.(pem|p12|pfx|jks|kdbx|age)$", re.IGNORECASE),
)

ALLOWLIST_KEY_PATTERNS = (
    re.compile(r"(^|_)TOKEN_ID$", re.IGNORECASE),
    re.compile(r"(^|_)(MARKET|EVENT|CONDITION|ASSET|CHAT|ORDER|REQUEST|COORDINATION|TRACE)_ID$", re.IGNORECASE),
    re.compile(r"(^|_)PUBLIC_KEY$", re.IGNORECASE),
)

SENSITIVE_KEY_PATTERNS = (
    re.compile(r"PRIVATE_KEY$", re.IGNORECASE),
    re.compile(r"API_KEY$", re.IGNORECASE),
    re.compile(r"API_SECRET$", re.IGNORECASE),
    re.compile(r"SECRET$", re.IGNORECASE),
    re.compile(r"TOKEN$", re.IGNORECASE),
    re.compile(r"BOT_TOKEN$", re.IGNORECASE),
    re.compile(r"PASSPHRASE$", re.IGNORECASE),
    re.compile(r"PASSWORD$", re.IGNORECASE),
)

ASSIGNMENT_RE = re.compile(
    r"(?P<quote>[\"']?)(?P<key>[A-Za-z_][A-Za-z0-9_]*)(?P=quote)\s*[:=]\s*(?P<value>.+)"
)
INLINE_KEY_VALUE_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\s|,;]+)")

STRING_LITERAL_RE = re.compile(r"[\"'](?P<value>[^\"'\n]{4,})[\"']")
PLACEHOLDER_RE = re.compile(
    r"\*\*\*REMOVED\*\*\*|REDACTED|REPLACE_ME|CHANGEME|EXAMPLE|DUMMY|FAKE|TOPSECRET|DEADBEEF|SECRETKEY|HIDDEN|<[^>]+>|\$\{[^}]+\}",
    re.IGNORECASE,
)

KNOWN_TEST_SECRET_VALUES = {
    "0x59c6995e998f97a5a0044966f0945382dbf59596e17f7b7b6b6d3d4d6f8d2f4c",
}

HIGH_CONFIDENCE_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key block"),
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"), "GitHub token"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "GitHub fine-grained token"),
    (re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"), "GitLab token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key"),
    (re.compile(r"\bASIA[0-9A-Z]{16}\b"), "AWS temporary access key"),
    (re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{20,}\b"), "Telegram bot token"),
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    message: str


def _git_paths(args: Sequence[str]) -> list[Path]:
    output = subprocess.check_output(args, cwd=ROOT, text=True, encoding="utf-8")
    return [ROOT / line for line in output.splitlines() if line.strip()]


def staged_paths() -> list[Path]:
    return _git_paths(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"])


def tracked_paths() -> list[Path]:
    return _git_paths(["git", "ls-files"])


def should_scan_path(path: Path) -> bool:
    if path.is_dir() or ".git" in path.parts:
        return False
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    if path.name.lower().startswith(".env"):
        return True
    return False


def has_forbidden_secret_filename(path: Path) -> bool:
    unix_path = path.relative_to(ROOT).as_posix() if path.is_absolute() else path.as_posix()
    lower_name = path.name.lower()
    if lower_name in ALLOWED_ENV_FILENAMES:
        return False
    return any(pattern.search(unix_path) for pattern in FORBIDDEN_FILENAME_PATTERNS)


def is_sensitive_key(key: str) -> bool:
    if any(pattern.search(key) for pattern in ALLOWLIST_KEY_PATTERNS):
        return False
    return any(pattern.search(key) for pattern in SENSITIVE_KEY_PATTERNS)


def _candidate_secrets(value: str) -> Iterable[str]:
    yield value.strip().strip(",")
    for match in STRING_LITERAL_RE.finditer(value):
        yield match.group("value").strip()


def looks_like_real_secret(key: str, value: str) -> bool:
    candidate = value.strip().strip(",")
    if not candidate:
        return False
    if PLACEHOLDER_RE.search(candidate):
        return False
    if re.fullmatch(r"[A-Z0-9_]+", candidate):
        return False
    if any(marker in candidate for marker in ("os.getenv(", "_env(", "settings.", "getenv(", "env.get(")):
        return False

    key_upper = key.upper()

    if key_upper.endswith("PRIVATE_KEY"):
        return bool(re.fullmatch(r"(?:0x)?[A-Fa-f0-9]{64}", candidate))
    if key_upper.endswith("BOT_TOKEN") or key_upper == "TELEGRAM_BOT_TOKEN":
        return bool(re.fullmatch(r"\d{8,12}:[A-Za-z0-9_-]{20,}", candidate))
    if key_upper.endswith("PASSWORD"):
        return len(candidate) >= 8
    if key_upper.endswith("PASSPHRASE"):
        return len(candidate) >= 12
    if key_upper.endswith("API_KEY"):
        return bool(
            re.fullmatch(r"[A-Fa-f0-9]{32}", candidate)
            or re.fullmatch(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", candidate, re.IGNORECASE)
            or re.fullmatch(r"[A-Za-z0-9_-]{20,}", candidate)
        )
    if key_upper.endswith("API_SECRET") or key_upper.endswith("SECRET"):
        return bool(re.fullmatch(r"[A-Za-z0-9_+/=-]{20,}", candidate))
    if key_upper.endswith("TOKEN"):
        return bool(re.fullmatch(r"[A-Za-z0-9_+/=-]{20,}", candidate))
    return False


def is_allowlisted_secret(path: Path, value: str) -> bool:
    rel_path = path.relative_to(ROOT).as_posix() if path.is_absolute() else path.as_posix()
    return rel_path.startswith("tests/") and value in KNOWN_TEST_SECRET_VALUES


def scan_text(text: str, path: Path) -> list[Finding]:
    findings: list[Finding] = []
    rel_path = path.relative_to(ROOT).as_posix() if path.is_absolute() else path.as_posix()

    if has_forbidden_secret_filename(path):
        findings.append(Finding(rel_path, 1, "tracked secret-like filename"))

    for lineno, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or "secret-scan: ignore" in line:
            continue

        assignment_matches: list[tuple[str, str]] = []
        match = ASSIGNMENT_RE.search(raw_line)
        if match and is_sensitive_key(match.group("key")):
            assignment_matches.append((match.group("key"), match.group("value")))
        for inline_match in INLINE_KEY_VALUE_RE.finditer(raw_line):
            key = inline_match.group("key")
            if is_sensitive_key(key):
                assignment_matches.append((key, inline_match.group("value")))

        key_finding_added = False
        for key, value in assignment_matches:
            for candidate in _candidate_secrets(value):
                if is_allowlisted_secret(path, candidate):
                    continue
                if looks_like_real_secret(key, candidate):
                    findings.append(Finding(rel_path, lineno, f"hardcoded value for {key}"))
                    key_finding_added = True
                    break
            if key_finding_added:
                break
        if key_finding_added:
            continue

        for pattern, label in HIGH_CONFIDENCE_SECRET_PATTERNS:
            if pattern.search(raw_line) and not PLACEHOLDER_RE.search(raw_line):
                findings.append(Finding(rel_path, lineno, f"{label} literal"))
                break

    return findings


def scan_paths(paths: Iterable[Path]) -> list[Finding]:
    findings: list[Finding] = []
    seen_paths: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen_paths or not path.exists() or not should_scan_path(path):
            continue
        seen_paths.add(resolved)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        findings.extend(scan_text(text, path))
    return findings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan tracked or staged files for likely hardcoded secrets.")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--tracked", action="store_true", help="Scan all tracked files.")
    scope.add_argument("--staged", action="store_true", help="Scan staged files only.")
    parser.add_argument("paths", nargs="*", help="Optional explicit paths to scan.")
    return parser.parse_args(argv)


def selected_paths(args: argparse.Namespace) -> list[Path]:
    if args.paths:
        return [(ROOT / path).resolve() if not Path(path).is_absolute() else Path(path) for path in args.paths]
    if args.staged:
        return staged_paths()
    return tracked_paths()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    findings = scan_paths(selected_paths(args))
    if not findings:
        print("Secret scan passed.")
        return 0

    print("Secret scan failed:")
    for finding in findings:
        print(f"- {finding.path}:{finding.line}: {finding.message}")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())