"""Validate downloaded predictions and logs without running model inference."""
from pathlib import Path
import ast
import hashlib
import json
import re
import sys
import numpy as np
import pandas as pd

SUB = Path(__file__).resolve().parents[1]
ROOT = next(p for p in SUB.parents if (p / 'eval.py').is_file())
sys.path.insert(0, str(ROOT))
import eval as ev


def verify():
    labels = ROOT / 'data/labels'
    frames = {s: pd.read_csv(labels / f'{s}_subset0.csv') for s in ('train', 'val', 'test')}
    sets = {s: set(f.Filename) for s, f in frames.items()}
    assert all(len(sets[s]) == len(frames[s]) for s in sets)
    assert not (sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test'])
    assert len(set.union(*sets.values())) == 17509
    all_labels = pd.read_csv(labels / 'labels.csv').set_index('Filename').Label
    label_differences = []
    for split, f in frames.items():
        reference = all_labels.loc[f.Filename].to_numpy()
        for i in np.flatnonzero(f.Label.to_numpy() != reference):
            label_differences.append({'split': split, 'Filename': f.iloc[i].Filename,
                                     'fold_label': int(f.iloc[i].Label), 'labels_csv_label': int(reference[i])})
    stats = json.loads((SUB / 'logs/split_stats.json').read_text())
    assert stats['n'] == {s: len(f) for s, f in frames.items()}
    for s, f in frames.items():
        assert stats['per_class'][s] == {str(k): int(v) for k, v in f.Label.value_counts().items()}
    verified, metrics = [], {}
    for file in sorted((SUB / 'predictions').glob('*.csv')):
        split = 'test' if file.stem.endswith('_test') else 'val'
        pred = ev.read_pred(str(file))
        ev.check_against_csv(pred, str(labels / f'{split}_subset0.csv'), split)
        m = ev.compute_metrics(pred.y_true, pred.y_pred, pred.probs)
        metrics[file.stem] = m
        verified.append({'file': file.name, 'n': len(pred.y_true),
                         'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                         **{k: float(m[k]) for k in ev.SCALARS}})
    for name in ('backbones', 'training', 'noise'):
        for r in json.loads((SUB / f'logs/{name}.json').read_text()):
            # F01 val files were replaced by the final multiscale + TS export;
            # noise.json records the training checkpoint's identity evaluation.
            if r['exp_id'] == 'F01':
                assert list((SUB / 'curves').glob(f"F01_*_seed{r['seed']}.png"))
                continue
            m = metrics[f"{r['exp_id']}_seed{r['seed']}_val"]
            for k in ('macro_f1', 'top1', 'ece'):
                assert abs(r[f'{k}_val'] - m[k]) < 1e-8, (name, r['exp_id'], k)
            assert list((SUB / 'curves').glob(f"{r['exp_id']}_*_seed{r['seed']}.png"))
    for r in json.loads((SUB / 'logs/inference.json').read_text()):
        m = metrics[f"{r['exp_id']}_seed0_val"]
        assert all(abs(r[f'{k}_val'] - m[k]) < 1e-8 for k in ('macro_f1', 'top1', 'ece'))
    for r in json.loads((SUB / 'logs/final.json').read_text()):
        m = metrics[f"{r['exp_id']}_seed{r['seed']}_test"]
        assert all(abs(r[f'{k}_test'] - m[k]) < 1e-8 for k in ev.SCALARS)
    for r in json.loads((SUB / 'logs/perclass.json').read_text()):
        m = metrics[f"{r['exp_id']}_seed{r['seed']}_test"]
        idx = [x.lower().replace(' ', '') for x in ev.load_names(str(labels / 'labels.csv'))].index(
            r['class'].lower().replace(' ', '').replace('negatives', 'negative'))
        assert all(abs(r[k] - m[k][idx]) < 1e-8 for k in ('precision','recall','f1','support'))
    notebook = json.loads((SUB / 'code/lab_day2.ipynb').read_text(encoding='utf-8'))
    history = {}
    for cell in notebook['cells']:
        for output in cell.get('outputs', []):
            for line in ''.join(output.get('text', [])).splitlines():
                match = re.match(r'^([BTF]\d+) (\d+) (\{.*\})$', line)
                if match:
                    entry = ast.literal_eval(match[3])
                    if 'epoch' in entry:
                        history.setdefault((match[1], int(match[2])), {})[entry['epoch']] = entry
    for (tag, seed), epochs in history.items():
        target = SUB / 'logs/history' / tag / f'seed{seed}'
        target.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([epochs[k] for k in sorted(epochs)]).to_csv(target / 'history.csv', index=False)
    for name in ('backbones', 'training', 'noise'):
        for r in json.loads((SUB / f'logs/{name}.json').read_text()):
            epochs = history[(r['exp_id'], r['seed'])]
            assert len(epochs) == r['epochs'], (r['exp_id'], r['seed'], 'incomplete history')
            best = max(epochs.values(), key=lambda x: (x['val_macro_f1'], -x['epoch']))
            assert best['epoch'] == r['best_epoch']
            assert abs(best['val_macro_f1'] - r['macro_f1_val']) < 1e-8
            assert abs(best['val_top1'] - r['top1_val']) < 1e-8
    result = {'predictions_verified': len(verified), 'curves': len(list((SUB / 'curves').glob('*.png'))),
              'split_counts': {s: len(f) for s, f in frames.items()}, 'union': 17509,
              'history_runs_recovered_from_notebook': len(history), 'label_differences': label_differences,
              'predictions': verified}
    (SUB / 'eval_out').mkdir(exist_ok=True)
    (SUB / 'eval_out/validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    names = ev.load_names(str(labels / 'labels.csv'))
    for tag, split in [('F01','test'), ('T00','test'), ('F01uncal','test'), ('F01','val'), ('T00','val')]:
        g = ev.load_group(str(SUB / f'predictions/{tag}_seed*_{split}.csv'), str(labels / f'{split}_subset0.csv'), ref_what=split)
        ev.save_group(SUB / 'eval_out', f'{tag}_{split}', g, names)
        print(ev.report_group(f'{tag}_{split}', g, names))
    print(f"Verified {len(verified)} prediction files; recovered {len(history)} histories.")


if __name__ == '__main__':
    verify()
