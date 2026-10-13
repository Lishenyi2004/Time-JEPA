"""Validate a staged release without loading datasets or checkpoints."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str):
                yield key, item
            else:
                yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def validate():
    errors = []
    manifest = json.loads((ROOT / 'release-manifest.json').read_text())
    policy = json.loads((ROOT / 'metadata-policy.json').read_text())
    local_only = {item['path'] for item in policy['restricted_metadata']}
    local_prefixes = tuple(policy['local_only_prefixes'])
    for record in manifest:
        if record['path'] in local_only or record['path'].startswith(local_prefixes + ('private_metadata/',)):
            errors.append(f'Local-only metadata selected for publication: {record["path"]}')
    published = [ROOT / record['path'] for record in manifest]
    python_files = sorted(path for path in published if path.suffix == '.py' and path.is_file())
    configs = sorted(path for path in published if path.suffix == '.yaml' and path.is_file())
    for path in python_files:
        try:
            compile(path.read_text(), str(path), 'exec')
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f'{path.relative_to(ROOT)}: {exc}')
    local_roots = {"src", "app", "covla_jepa_rjob", "covla_probe_rjob", "covla_law_rjob", "grasp_eval_rjob", "physion_ocp_rjob", "intphys2_eval_rjob", "world_model_probe_rjob", "physics_consistency", "external_baselines"}
    for path in python_files:
        relative = str(path.relative_to(ROOT))
        package = relative[:-3].replace("/", ".").rpartition(".")[0]
        if relative.endswith("/__init__.py"):
            package = relative[:-12].replace("/", ".")
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split(".")
                    prefix = ".".join(parts[:len(parts) - node.level + 1])
                    modules = [".".join(item for item in [prefix, node.module] if item)]
                else:
                    modules = [node.module or ""]
            else:
                continue
            for module in modules:
                if module.split(".")[0] not in local_roots:
                    continue
                target = ROOT / module.replace(".", "/")
                if not target.is_dir() and not target.with_suffix(".py").is_file():
                    errors.append(f"{relative}: missing local import {module}")
    for path in configs:
        try:
            config = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            errors.append(f'{path.relative_to(ROOT)}: {exc}')
            continue
        if not isinstance(config, dict):
            errors.append(f'{path.relative_to(ROOT)}: expected a YAML mapping')
            continue
        for key in ['base', 'base_config']:
            value = config.get(key)
            if value and not any(p.is_file() for p in [path.parent / value, ROOT / value]):
                errors.append(f'{path.relative_to(ROOT)}: missing {key} {value}')
        for key, value in strings(config):
            if '/mnt/' in value or '/root/' in value:
                errors.append(f'{path.relative_to(ROOT)}: machine-local path in {key}')
    for record in manifest:
        path = ROOT / record['path']
        if not path.is_file():
            errors.append(f'Missing selected file: {record["path"]}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != record['release_sha256']:
            errors.append(f'Checksum mismatch: {record["path"]}')
    aliases = json.loads((ROOT / 'checkpoint-aliases.json').read_text())
    for record in aliases:
        path = ROOT / record['config']
        if record['release_checkpoint'] not in path.read_text():
            errors.append(f'Checkpoint alias missing: {record["config"]}')
        value = record['release_checkpoint']
        expected = '/last/transformer/model.safetensors' if record['key'] == 'adapted_transformer_checkpoint' else None
        if expected and not value.endswith(expected):
            errors.append(f'Invalid iVideoGPT alias: {value}')
        elif not expected and not Path(value).name.startswith('last.'):
            errors.append(f'Invalid encoder alias: {value}')
    forbidden = ('.pt', '.pth', '.pth.tar', '.ckpt', '.safetensors', '.mp4', '.avi', '.parquet', '.h5', '.hdf5', '.zip', '.urdf')
    for path in published:
        if path.is_file() and path.name.endswith(forbidden):
            errors.append(f'Excluded resource copied: {path.relative_to(ROOT)}')
    report = {'python_files': len(python_files), 'yaml_files': len(configs), 'encoder_aliases': len(aliases), 'errors': errors}
    print(json.dumps(report, indent=2))
    return not errors


if __name__ == '__main__':
    raise SystemExit(0 if validate() else 1)
