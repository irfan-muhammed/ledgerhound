"""
ledgerhound.tools.config_tools
===========================

Shell tools that let the AGENT manage the unified project config: when a
user pastes a Jira token / repo URL / MinIO endpoint in chat, the model
calls ``configure_integration`` and Ledgerhound files it correctly -- secret
values into ``.ledgerhound/secrets.env``, everything else into
``.ledgerhound/config.yaml``. ``show_config`` renders a masked summary.
"""

from __future__ import annotations

from ledgerhound.project.settings import SECTIONS, ProjectSettings
from ledgerhound.tools.registry import ToolDef

# L-04: matched on the key's own words, not any substring "key" -- the old
# rule filed jira.project_key (a plain setting like "AUD") into secrets.env,
# so the config never had it and jira_create_issue always failed.
_SECRET_WORDS = {"token", "password", "passwd", "secret", "apikey", "api_key",
                 "secret_key", "access_key", "pat", "credential", "credentials"}
_NOT_SECRET = {"project_key", "issue_key", "key_env", "token_env"}


def _looks_secret(key: str) -> bool:
    k = key.lower().strip()
    if k in _NOT_SECRET or k.endswith("_env"):
        return False
    if k in _SECRET_WORDS:
        return True
    parts = k.replace("-", "_").split("_")
    return any(p in {"token", "password", "passwd", "secret", "pat"} for p in parts) \
        or k.endswith(("api_key", "secret_key", "access_key"))


def redact_args(tool: str, args: dict) -> dict:
    """L-05: the copy of a tool call's arguments that is safe to LOG --
    transcript.jsonl and the hash-chained events.jsonl are permanent, so a
    secret written there can never be removed."""
    if tool != "configure_integration" or not isinstance(args, dict):
        return args
    if args.get("is_secret") or _looks_secret(str(args.get("key", ""))):
        return {**args, "value": "***redacted***"}
    return args


def build_config_tools(settings: ProjectSettings) -> list[ToolDef]:
    def configure_integration(section: str, key: str, value: str,
                              is_secret: bool = False) -> str:
        section = section.strip().lower()
        if section not in SECTIONS:
            return f"error: unknown section '{section}'; one of {', '.join(SECTIONS)}"
        if section == "minio":
            # L-06: the team object store is main_config.yaml `minio:` plus
            # the shell's /minio command; the project file is not read.
            return ("not stored: the object store is configured in "
                    "main_config.yaml (`minio:` endpoint/bucket/access_key) and "
                    "its secret with the shell command /minio --key, which "
                    "checks it and never sends it to the model. Tell the user "
                    "to run /minio --key -- and not to paste secrets into chat.")
        if is_secret or _looks_secret(key):
            env_name = settings.set_secret(section, key, value)
            return (f"stored secret for {section}.{key}: value -> "
                    f"{settings.secrets_path.name} (gitignored), config records "
                    f"env name {env_name}. It is active in this session already.")
        settings.set_value(section, key, value)
        return f"set {section}.{key} = {value} in {settings.config_path}"

    def show_config() -> str:
        return (f"config file: {settings.config_path}\n"
                f"secrets file: {settings.secrets_path} (values never shown)\n\n"
                + settings.masked_summary())

    def _confirm(args: dict) -> str:
        key = args.get("key", "?")
        sec = args.get("section", "?")
        if args.get("is_secret") or _looks_secret(str(key)):
            return (f"Store a SECRET for {sec}.{key} into secrets.env "
                    f"(value hidden)?")
        return f"Set {sec}.{key} = {str(args.get('value'))[:60]} in config.yaml?"

    return [
        ToolDef(
            name="configure_integration",
            description=(
                "Save a configuration value the user gave you (Jira URL/email/"
                "token, git remote/branch/token, MinIO endpoint/keys, Trino "
                "host/creds, LLM/judge config ref) into the project's unified "
                "config. Secrets are stored safely in secrets.env, never in "
                "the yaml. Sections: llm, judge, jira, git, minio, trino."),
            parameters={"type": "object", "properties": {
                "section": {"type": "string",
                            "enum": list(SECTIONS)},
                "key": {"type": "string",
                        "description": "e.g. base_url, email, project_key, "
                                       "token, remote, default_branch, endpoint, host"},
                "value": {"type": "string"},
                "is_secret": {"type": "boolean",
                              "description": "true for tokens/passwords/keys"}},
                "required": ["section", "key", "value"]},
            fn=configure_integration, hitl="confirm", confirm_prompt=_confirm),
        ToolDef(
            name="show_config",
            description="Show the project's integration config (secret values "
                        "masked; shows whether each secret resolves).",
            parameters={"type": "object", "properties": {}, "required": []},
            fn=show_config),
    ]
