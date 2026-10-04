#!/usr/bin/env python3
"""Verify the code-and-datasets distribution without historical traces or services."""
from __future__ import annotations
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
from semantic_guided_kbqa.config import AppConfig
from semantic_guided_kbqa.data import TrainingContract


def main():
    configs = sorted((ROOT / 'configs').glob('pipeline*.json'))
    if not configs:
        raise FileNotFoundError('No pipeline configurations')
    for path in configs:
        config = AppConfig.load(path)
        config.validate()
        data = config.data
        TrainingContract.load(data.semantic_train, data.compose_train, data.operator_train)
        for name in ('decompose_predictions', 'gold_entities'):
            source = Path(getattr(data, name))
            if not source.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError(f'{path.name}: external dataset path: {name}')
            if not json.loads(source.read_text(encoding='utf-8')):
                raise ValueError(f'{path.name}: empty dataset: {name}')
    counts = {}
    for path in sorted((ROOT / 'data').rglob('*.json')):
        value = json.loads(path.read_text(encoding='utf-8'))
        if not value:
            raise ValueError(f'Empty data: {path}')
        counts[str(path.relative_to(ROOT))] = len(value)
    for path in sorted((ROOT / 'data/indexes').glob('*.json.gz')):
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            if not json.load(stream):
                raise ValueError(f'Empty retrieval index: {path}')
    for name in ('fb_roles', 'fb_types', 'reverse_properties', 'domain_info', 'domain_dict'):
        if not (ROOT / 'ontology' / name).is_file():
            raise FileNotFoundError(name)
    if any(path.is_symlink() for path in ROOT.rglob('*')):
        raise ValueError('Distribution contains symlinks')
    print(json.dumps({'status': 'ok', 'configs': len(configs), 'dataset_rows': counts}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
