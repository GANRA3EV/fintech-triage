import argparse
import json
from pathlib import Path

import pandas as pd

from triage.experiment import execute, write_json
from triage.resources import Resources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--backend', required=True)
    args = parser.parse_args()
    metadata = json.loads(Path(args.metadata).read_text(encoding='utf-8'))
    parts = [pd.read_excel(args.data, sheet_name=name, dtype={'client_id': str})
             for name in ('train', 'validation', 'test')]
    with Resources() as resources:
        execute(parts, metadata['categories'], args.output, args.backend, metadata['provenance'])
    write_json(Path(args.output) / 'resources.json', {
        'process_seconds': resources.elapsed, 'sampled_peak_rss_bytes': resources.peak,
        'baseline_rss_bytes': resources.baseline, 'sampling_interval_seconds': 0.01,
        'scope': 'Текущий процесс; импорт и чтение XLSX до измерения; дочерние процессы не учтены'})


if __name__ == '__main__':
    main()
