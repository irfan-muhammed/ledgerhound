"""
ledgerhound.project.settings
=========================

One config for everything -- ``<project>/.ledgerhound/config.yaml`` holds the
llm / jira / git / minio / trino sections together, heavily commented so a
new user can fill it in without reading docs. Secrets NEVER live in the
yaml: the yaml stores env-var *names*; the values go to
``.ledgerhound/secrets.env`` (auto-gitignored) or the OS environment.

The agent itself can write this file: the ``configure_integration`` shell
tool routes here, so a user can paste a token in chat and Ledgerhound files it
correctly (value -> secrets.env, env-name -> config.yaml).

Merge order when reading a setting: OS env > secrets.env > project yaml.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_NAME = "config.yaml"
SECRETS_NAME = "secrets.env"

SECTIONS = ("llm", "judge", "results", "jira", "git", "minio", "trino")

# default env-var name per (section, key) when the user stores a secret
DEFAULT_SECRET_ENVS = {
    ("jira", "token"): "LEDGERHOUND_JIRA_TOKEN",
    ("git", "token"): "LEDGERHOUND_GIT_TOKEN",
    ("minio", "access_key"): "MINIO_ACCESS_KEY",
    ("minio", "secret_key"): "MINIO_SECRET_KEY",
    ("trino", "password"): "LEDGERHOUND_TRINO_PASSWORD",
    ("llm", "api_key"): "LEDGERHOUND_LLM_API_KEY",
    ("judge", "api_key"): "LEDGERHOUND_JUDGE_API_KEY",
}

_TEMPLATE = """\
# =============================================================================
# Ledgerhound project configuration -- ONE file for every integration.
#
#   How to fill this in (or just paste tokens in the chat and Ledgerhound will
#   file them here for you):
#
#   * Plain settings (URLs, emails, branch names) go right here.
#   * SECRETS (tokens, passwords, keys) do NOT go in this file. Put the value
#     in `.ledgerhound/secrets.env` (KEY=value, one per line -- gitignored) or in
#     your OS environment. This file only names WHICH env var to read.
#
#   Sections you don't use can stay empty -- Ledgerhound only enables an
#   integration when its required fields are set.
# =============================================================================

llm:
  # Model endpoint. `config_ref` points at a full LLM yaml (e.g.
  # configs/deepseek.yaml); leave empty to pass --llm-config on the CLI.
  config_ref: "{llm_config_ref}"
  # Name of the env var the chosen config's own `api_key_env` names (value
  # goes in secrets.env!). Usually set automatically when config_ref is set.
  api_key_env: "{llm_api_key_env}"

judge:
  # Optional SECOND model for evaluation/judging. Same yaml shape as `llm`
  # (e.g. configs/deepseek.yaml). Set via /model-judge in the shell.
  config_ref: "{judge_config_ref}"
  api_key_env: "{judge_api_key_env}"
  # Override just the MODEL id on top of config_ref/self-judge's endpoint
  # -- set via /model-judge <model-id> (no config_ref change, applies on
  # the next judge resolution). Blank = use whatever config_ref/self-judge
  # already resolves to.
  model: "{judge_model}"

results:
  # Where analysis runs write their output folder (plan.json, findings.jsonl,
  # events.jsonl, charts, the exported notebook, ...). Leave empty to use the
  # built-in default (<project>/run_outputs).
  output_dir: "{results_output_dir}"

jira:
  # Your Jira Cloud/Server base URL, e.g. https://yourorg.atlassian.net
  base_url: "{jira_base_url}"
  # The email you log into Jira with (Cloud API tokens pair with email).
  email: "{jira_email}"
  # Project key for new issues, e.g. AUD or DATA.
  project_key: "{jira_project_key}"
  # Name of the env var holding your API token (value goes in secrets.env!)
  token_env: "{jira_token_env}"
  # "3" for Jira Cloud, "2" for older Jira Server.
  api_version: "{jira_api_version}"
  # basic = email + API token (Jira Cloud); bearer = personal access token
  # (Jira Server / Data Center).
  auth: "{jira_auth}"
  # "false" only for an internal Jira whose certificate this machine does
  # not trust (e.g. an org-internal CA).
  verify_ssl: "{jira_verify_ssl}"

git:
  # Remote to push/pull, e.g. origin or a full URL.
  remote: "{git_remote}"
  # Branch to sync when you say "push my work".
  default_branch: "{git_default_branch}"
  # Optional env var holding a token for https pushes (value in secrets.env).
  token_env: "{git_token_env}"

minio:
  # Artifact store, e.g. minio.internal:9000 -- leave empty to disable.
  endpoint: "{minio_endpoint}"
  access_key_env: "{minio_access_key_env}"
  secret_key_env: "{minio_secret_key_env}"

trino:
  # Big-data push-down engine -- leave host empty to run local-only.
  host: "{trino_host}"
  port: "{trino_port}"
  catalog: "{trino_catalog}"
  schema: "{trino_schema}"
  user: "{trino_user}"
  password_env: "{trino_password_env}"
"""

_DEFAULTS = {
    "llm": {"config_ref": "", "api_key_env": ""},
    "judge": {"config_ref": "", "api_key_env": "", "model": ""},
    "results": {"output_dir": ""},
    "jira": {"base_url": "", "email": "", "project_key": "",
             "token_env": "LEDGERHOUND_JIRA_TOKEN", "api_version": "3",
             "auth": "basic", "verify_ssl": "true"},
    "git": {"remote": "origin", "default_branch": "main",
            "token_env": ""},
    "minio": {"endpoint": "", "access_key_env": "MINIO_ACCESS_KEY",
              "secret_key_env": "MINIO_SECRET_KEY"},
    "trino": {"host": "", "port": "8080", "catalog": "", "schema": "",
              "user": "", "password_env": "LEDGERHOUND_TRINO_PASSWORD"},
}

_SECRET_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _yaml_dq_escape(value: str) -> str:
    """Every field in _TEMPLATE sits inside a double-quoted YAML scalar --
    a raw Windows path (`C:\\Users\\...`) written in unescaped breaks the
    YAML (backslash starts an escape sequence there) and previously got
    silently swallowed by `load()`'s bare `except`, wiping the whole config
    back to defaults on next read. Escape backslash and double-quote so any
    value round-trips regardless of what's in it."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


@dataclass
class ProjectSettings:
    """Read/write view over .ledgerhound/config.yaml + secrets.env."""
    dotdir: Path
    data: dict = field(default_factory=dict)

    # -- construction --------------------------------------------------------
    @classmethod
    def load(cls, project_or_dotdir) -> "ProjectSettings":
        dotdir = getattr(project_or_dotdir, "dotdir", None) or Path(project_or_dotdir)
        s = cls(dotdir=Path(dotdir))
        path = s.config_path
        data = {}
        if path.exists():
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except Exception:
                data = {}
        # deep-merge over defaults so missing keys never KeyError
        merged = {}
        for sec, defaults in _DEFAULTS.items():
            merged[sec] = {**defaults, **(data.get(sec) or {})}
        s.data = merged
        return s

    @property
    def config_path(self) -> Path:
        return self.dotdir / CONFIG_NAME

    @property
    def secrets_path(self) -> Path:
        return self.dotdir / SECRETS_NAME

    # -- read -----------------------------------------------------------------
    def get(self, section: str, key: str, default: str = "") -> str:
        return str(self.data.get(section, {}).get(key, default) or default)

    def section(self, name: str) -> dict:
        return dict(self.data.get(name, {}))

    def secret(self, section: str, key_env_field: str = "token_env") -> str:
        """Resolve a secret value: OS env first, then secrets.env."""
        env_name = self.get(section, key_env_field)
        if not env_name:
            return ""
        return os.environ.get(env_name) or self._secrets_map().get(env_name, "")

    # -- write ------------------------------------------------------------------
    def set_value(self, section: str, key: str, value: str) -> None:
        if section not in SECTIONS:
            raise ValueError(f"unknown section {section!r}; one of {SECTIONS}")
        self.data.setdefault(section, {})[key] = value
        self.save()

    def set_secret(self, section: str, key: str, value: str,
                   env_name: str = "") -> str:
        """Store a secret VALUE in secrets.env and record its env NAME in the
        yaml (`<key>_env`). Returns the env name used."""
        if section not in SECTIONS:
            raise ValueError(f"unknown section {section!r}; one of {SECTIONS}")
        env_name = env_name or DEFAULT_SECRET_ENVS.get(
            (section, key), f"LEDGERHOUND_{section.upper()}_{key.upper()}")
        if not _SECRET_KEY_RE.match(env_name):
            raise ValueError(f"invalid env var name {env_name!r}")
        secrets = self._secrets_map()
        secrets[env_name] = value
        self._write_secrets(secrets)
        env_field = key if key.endswith("_env") else f"{key}_env"
        self.data.setdefault(section, {})[env_field] = env_name
        self.save()
        # make it visible to this very process immediately
        os.environ[env_name] = value
        return env_name

    def save(self) -> None:
        self.dotdir.mkdir(parents=True, exist_ok=True)
        self._ensure_gitignore()
        fields = {}
        for sec, defaults in _DEFAULTS.items():
            for k, dflt in defaults.items():
                fields[f"{sec}_{k}"] = _yaml_dq_escape(self.get(sec, k, str(dflt)))
        # llm section uses a single key with a different template name
        fields["llm_config_ref"] = _yaml_dq_escape(self.get("llm", "config_ref"))
        self.config_path.write_text(_TEMPLATE.format(**fields), encoding="utf-8")

    # -- secrets.env ------------------------------------------------------------
    def _secrets_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.secrets_path.exists():
            for line in self.secrets_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
        return out

    def _write_secrets(self, secrets: dict[str, str]) -> None:
        self.dotdir.mkdir(parents=True, exist_ok=True)
        self._ensure_gitignore()
        lines = ["# Ledgerhound secrets -- NEVER commit this file (auto-gitignored).",
                 "# KEY=value, one per line. Referenced by name from config.yaml."]
        lines += [f"{k}={v}" for k, v in sorted(secrets.items())]
        self.secrets_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            # Owner read/write only -- plaintext credentials on disk should
            # not be group/world-readable. No-op on Windows (ACLs, not POSIX
            # mode bits, govern access there) but real on Linux/macOS deploys.
            os.chmod(self.secrets_path, 0o600)
        except OSError:
            pass

    def load_secrets_env(self) -> int:
        """Export secrets.env into os.environ (no overwrite). Returns count set."""
        n = 0
        for k, v in self._secrets_map().items():
            if k not in os.environ:
                os.environ[k] = v
                n += 1
        return n

    def _ensure_gitignore(self) -> None:
        gi = self.dotdir / ".gitignore"
        want = ("secrets.env", "sessions/")
        existing = gi.read_text(encoding="utf-8").splitlines() if gi.exists() else []
        missing = [w for w in want if w not in existing]
        if missing:
            gi.write_text("\n".join(existing + missing) + "\n", encoding="utf-8")

    # -- display -------------------------------------------------------------------
    def masked_summary(self) -> str:
        """Human-readable summary; secret VALUES never shown, only env names +
        whether they resolve."""
        lines = []
        for sec in SECTIONS:
            if sec == "minio":
                # The team object store is configured once per user
                # (main_config.yaml `minio:` + /minio --key), not per
                # project; this legacy block is read by nothing and its
                # "MISSING" rows contradicted the real store's status.
                lines.append("[minio] team object store -- see the "
                             "[object store] line below, or /minio")
                continue
            body = self.section(sec)
            if not any(str(v).strip() for v in body.values()):
                lines.append(f"[{sec}] (not configured)")
                continue
            lines.append(f"[{sec}]")
            for k, v in body.items():
                if k.endswith("_env"):
                    status = "set" if (v and (os.environ.get(str(v))
                                              or self._secrets_map().get(str(v)))) \
                        else "MISSING"
                    lines.append(f"  {k} = {v or '(unset)'}  (value: {status})")
                else:
                    lines.append(f"  {k} = {v}")
        return "\n".join(lines)
