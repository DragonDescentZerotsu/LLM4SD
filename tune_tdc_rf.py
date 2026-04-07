#!/usr/bin/env python3
"""Tune RandomForest hyperparameters for TDC tasks using registered feature backends."""

from __future__ import annotations

import os
import argparse
import json
import math
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold

from feature_backends import load_feature_backend, prepare_feature_matrices

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable


def configure_single_thread_runtime():
    thread_limits = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in thread_limits.items():
        os.environ.setdefault(key, value)


def configure_worker_runtime():
    configure_single_thread_runtime()
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
    except Exception:
        pass


configure_single_thread_runtime()


_FEATURE_BACKEND = None
_FEATURE_TEMPLATE = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subtask', type=str, default='BBB_Martins', help='TDC task name')
    parser.add_argument('--feature_backend', type=str, default='bbb_martins', help='registered feature backend name, for example bbb_martins, dili, clintox, herg, pampa_ncats, skin_reaction, pgp_broccatelli, carcinogens_lagunin, ames, bioavailability_ma, hia_hou, cyp2c9_substrate_carbonmangels, cyp2d6_substrate_carbonmangels, cyp3a4_substrate_carbonmangels, sarscov2_3clpro_diamond, or sarscov2_vitro_touret')
    parser.add_argument('--precomputed_feature_dir', type=str, default='scaffold_datasets/TDC_precomputed_features', help='directory for cached precomputed TDC features')
    parser.add_argument('--feature_jobs', type=int, default=4, help='number of worker processes used for feature precomputation')
    parser.add_argument('--rf_jobs', type=int, default=1, help='number of CPU workers used by RandomForest itself during tuning')
    parser.add_argument('--force_recompute_features', action='store_true', help='recompute cached features even if they already exist')
    parser.add_argument('--n_iter', type=int, default=40, help='number of hyperparameter candidates to evaluate')
    parser.add_argument('--search_seed', type=int, default=0, help='random seed used to sample hyperparameter candidates')
    parser.add_argument('--eval_seeds', type=str, default='0,1,2,3,4', help='comma-separated RF random seeds used for scoring each candidate')
    parser.add_argument('--output_dir', type=str, default='eval_result/TDC_hyperparameter_search', help='directory to save search results')
    parser.add_argument('--selection_metric', choices=['macro_f1', 'roc_auc'], default='macro_f1', help='metric used to choose the best hyperparameter set')
    parser.add_argument(
        '--selection_mode',
        choices=['valid', 'train_cv_5fold'],
        default='valid',
        help=(
            "how to choose hyperparameters: 'valid' keeps the current train->valid selection flow; "
            "'train_cv_5fold' tunes only on train with 5-fold CV, then retrains on full train and reports valid"
        ),
    )
    return parser.parse_args()


def parse_seed_list(seed_text: str) -> list[int]:
    seeds = [int(item.strip()) for item in seed_text.split(',') if item.strip()]
    if not seeds:
        raise ValueError("eval_seeds must contain at least one integer")
    return seeds


def get_tdc_split_path(subtask: str, split_name: str) -> Path:
    path = Path('scaffold_datasets') / 'TDC' / subtask / f'{subtask}_{split_name}.csv'
    if not path.exists():
        raise FileNotFoundError(f"Missing TDC split file: {path}")
    return path


def load_tdc_split(subtask: str, split_name: str):
    path = get_tdc_split_path(subtask, split_name)
    df = pd.read_csv(path)
    return df['smiles'].astype(str).tolist(), df[subtask].tolist()


def init_feature_backend_worker(feature_backend_name: str):
    global _FEATURE_BACKEND, _FEATURE_TEMPLATE
    configure_worker_runtime()
    backend = load_feature_backend(feature_backend_name)
    _FEATURE_BACKEND = backend
    _FEATURE_TEMPLATE = {name: math.nan for name in backend.get_feature_names()}


def featurize_backend_smiles(smiles: str):
    row = dict(_FEATURE_TEMPLATE)
    try:
        row.update(_FEATURE_BACKEND.featurize_smiles(smiles))
    except Exception as exc:
        print(f"Feature precompute failed for SMILES {smiles!r}: {str(exc)}")
    row['smiles'] = smiles
    return row


def get_precomputed_feature_path(subtask: str, split_name: str, feature_backend: str, precomputed_feature_dir: str) -> Path:
    cache_dir = Path(precomputed_feature_dir) / subtask
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'{subtask}_{split_name}_{feature_backend}_features.csv'


def load_precomputed_feature_frame(feature_path: Path, smiles_list: list[str]) -> pd.DataFrame:
    df = pd.read_csv(feature_path)
    if 'smiles' not in df.columns:
        raise ValueError(f"Precomputed feature file is missing 'smiles' column: {feature_path}")
    cached_smiles = df['smiles'].astype(str).tolist()
    if cached_smiles != [str(smiles) for smiles in smiles_list]:
        raise ValueError(f"Cached features in {feature_path} do not align with dataset split order")
    return df.drop(columns=['smiles'])


def compute_and_cache_backend_features(
    smiles_list: list[str],
    subtask: str,
    split_name: str,
    feature_backend: str,
    precomputed_feature_dir: str,
    feature_jobs: int,
) -> pd.DataFrame:
    feature_path = get_precomputed_feature_path(subtask, split_name, feature_backend, precomputed_feature_dir)
    worker_count = max(1, feature_jobs)
    desc = f'TDC/{subtask} {split_name}'
    print(f'Precomputing {len(smiles_list)} molecules for {desc} using {worker_count} workers')

    if worker_count == 1:
        init_feature_backend_worker(feature_backend)
        rows = [featurize_backend_smiles(smiles) for smiles in tqdm(smiles_list, total=len(smiles_list), desc=desc)]
    else:
        chunksize = max(1, len(smiles_list) // (worker_count * 4))
        with multiprocessing.Pool(
            worker_count,
            initializer=init_feature_backend_worker,
            initargs=(feature_backend,),
        ) as pool:
            iterator = pool.imap(featurize_backend_smiles, smiles_list, chunksize=chunksize)
            rows = list(tqdm(iterator, total=len(smiles_list), desc=desc))

    feature_df = pd.DataFrame(rows)
    ordered_columns = ['smiles'] + [column for column in feature_df.columns if column != 'smiles']
    feature_df = feature_df[ordered_columns]
    feature_df.to_csv(feature_path, index=False)
    print(f'Saved precomputed features to {feature_path}')
    return feature_df.drop(columns=['smiles'])


def get_tdc_feature_frame(
    smiles_list: list[str],
    subtask: str,
    split_name: str,
    feature_backend: str,
    precomputed_feature_dir: str,
    feature_jobs: int,
    force_recompute_features: bool,
) -> pd.DataFrame:
    feature_path = get_precomputed_feature_path(subtask, split_name, feature_backend, precomputed_feature_dir)
    if feature_path.exists() and not force_recompute_features:
        print(f'Loading cached features from {feature_path}')
        return load_precomputed_feature_frame(feature_path, smiles_list)
    return compute_and_cache_backend_features(
        smiles_list,
        subtask,
        split_name,
        feature_backend,
        precomputed_feature_dir,
        feature_jobs,
    )


def get_parameter_space():
    return {
        'n_estimators': [100, 150, 200, 300, 400, 500, 700],
        'max_depth': [None, 4, 8, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10, 20, 50, 100],
        'min_samples_leaf': [1, 2, 3, 4, 5, 10],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [None, 'balanced', 'balanced_subsample'],
    }


def compute_macro_f1(y_true, y_pred):
    if len(set(y_true)) <= 1:
        return 0.0
    return f1_score(y_true, y_pred, average='macro')


def compute_roc_auc(y_true, y_score):
    if len(set(y_true)) <= 1:
        return math.nan
    return roc_auc_score(y_true, y_score)


def evaluate_candidate(params, X_train, y_train, X_valid, y_valid, seeds: list[int], rf_jobs: int):
    valid_roc_auc_scores = []
    train_roc_auc_scores = []
    valid_macro_f1_scores = []
    train_macro_f1_scores = []
    for seed in seeds:
        model = RandomForestClassifier(
            **params,
            random_state=seed,
            n_jobs=rf_jobs,
        )
        model.fit(X_train, y_train)
        train_proba = model.predict_proba(X_train)[:, 1]
        valid_proba = model.predict_proba(X_valid)[:, 1]
        train_pred = model.predict(X_train)
        valid_pred = model.predict(X_valid)
        train_roc_auc_scores.append(compute_roc_auc(y_train, train_proba))
        valid_roc_auc_scores.append(compute_roc_auc(y_valid, valid_proba))
        train_macro_f1_scores.append(compute_macro_f1(y_train, train_pred))
        valid_macro_f1_scores.append(compute_macro_f1(y_valid, valid_pred))
    return {
        'mean_train_roc_auc': float(np.nanmean(train_roc_auc_scores)),
        'std_train_roc_auc': float(np.nanstd(train_roc_auc_scores)),
        'mean_valid_roc_auc': float(np.nanmean(valid_roc_auc_scores)),
        'std_valid_roc_auc': float(np.nanstd(valid_roc_auc_scores)),
        'mean_train_macro_f1': float(np.nanmean(train_macro_f1_scores)),
        'std_train_macro_f1': float(np.nanstd(train_macro_f1_scores)),
        'mean_valid_macro_f1': float(np.nanmean(valid_macro_f1_scores)),
        'std_valid_macro_f1': float(np.nanstd(valid_macro_f1_scores)),
    }


def evaluate_candidate_with_train_cv(
    params,
    train_feature_frame: pd.DataFrame,
    y_train,
    *,
    seeds: list[int],
    rf_jobs: int,
    scale_features: bool,
    cv_n_splits: int,
    cv_seed: int,
):
    y_train_array = np.asarray(y_train, dtype=int)
    unique_labels, label_counts = np.unique(y_train_array, return_counts=True)
    if len(unique_labels) < 2:
        raise ValueError('train_cv_5fold requires at least two classes in the train split')

    min_class_count = int(label_counts.min())
    if min_class_count < cv_n_splits:
        raise ValueError(
            f"Cannot run {cv_n_splits}-fold CV because the smallest class in train has only "
            f"{min_class_count} samples."
        )

    splitter = StratifiedKFold(
        n_splits=cv_n_splits,
        shuffle=True,
        random_state=cv_seed,
    )

    valid_roc_auc_scores = []
    train_roc_auc_scores = []
    valid_macro_f1_scores = []
    train_macro_f1_scores = []

    for fold_train_idx, fold_valid_idx in splitter.split(train_feature_frame, y_train_array):
        fold_train_df = train_feature_frame.iloc[fold_train_idx]
        fold_valid_df = train_feature_frame.iloc[fold_valid_idx]
        x_fold_train, x_fold_valid, _, _ = prepare_feature_matrices(
            fold_train_df,
            fold_valid_df,
            fold_valid_df,
            scale_features=scale_features,
        )
        y_fold_train = y_train_array[fold_train_idx]
        y_fold_valid = y_train_array[fold_valid_idx]

        for seed in seeds:
            model = RandomForestClassifier(
                **params,
                random_state=seed,
                n_jobs=rf_jobs,
            )
            model.fit(x_fold_train, y_fold_train)

            train_proba = model.predict_proba(x_fold_train)[:, 1]
            valid_proba = model.predict_proba(x_fold_valid)[:, 1]
            train_pred = model.predict(x_fold_train)
            valid_pred = model.predict(x_fold_valid)

            train_roc_auc_scores.append(compute_roc_auc(y_fold_train, train_proba))
            valid_roc_auc_scores.append(compute_roc_auc(y_fold_valid, valid_proba))
            train_macro_f1_scores.append(compute_macro_f1(y_fold_train, train_pred))
            valid_macro_f1_scores.append(compute_macro_f1(y_fold_valid, valid_pred))

    return {
        'mean_train_roc_auc': float(np.nanmean(train_roc_auc_scores)),
        'std_train_roc_auc': float(np.nanstd(train_roc_auc_scores)),
        'mean_valid_roc_auc': float(np.nanmean(valid_roc_auc_scores)),
        'std_valid_roc_auc': float(np.nanstd(valid_roc_auc_scores)),
        'mean_train_macro_f1': float(np.nanmean(train_macro_f1_scores)),
        'std_train_macro_f1': float(np.nanstd(train_macro_f1_scores)),
        'mean_valid_macro_f1': float(np.nanmean(valid_macro_f1_scores)),
        'std_valid_macro_f1': float(np.nanstd(valid_macro_f1_scores)),
    }


def extract_best_params(record):
    keys = [
        'n_estimators',
        'max_depth',
        'min_samples_split',
        'min_samples_leaf',
        'bootstrap',
        'criterion',
        'max_features',
        'class_weight',
    ]
    return {key: record[key] for key in keys}


def main():
    args = parse_args()
    load_feature_backend(args.feature_backend)

    eval_seeds = parse_seed_list(args.eval_seeds)
    train_smiles, y_train = load_tdc_split(args.subtask, 'train')
    valid_smiles, y_valid = load_tdc_split(args.subtask, 'valid')

    train_df = get_tdc_feature_frame(
        train_smiles,
        args.subtask,
        'train',
        args.feature_backend,
        args.precomputed_feature_dir,
        args.feature_jobs,
        args.force_recompute_features,
    )
    valid_df = get_tdc_feature_frame(
        valid_smiles,
        args.subtask,
        'valid',
        args.feature_backend,
        args.precomputed_feature_dir,
        args.feature_jobs,
        args.force_recompute_features,
    )

    scale_features = True
    X_train, X_valid, _, surviving_columns = prepare_feature_matrices(
        train_df,
        valid_df,
        valid_df,
        scale_features=scale_features,
    )
    print(f'Using {len(surviving_columns)} surviving features for tuning')

    candidate_params = list(ParameterSampler(get_parameter_space(), n_iter=args.n_iter, random_state=args.search_seed))
    results = []
    best_record = None
    selection_mean_key = f'mean_valid_{args.selection_metric}'
    selection_std_key = f'std_valid_{args.selection_metric}'

    for index, params in enumerate(tqdm(candidate_params, total=len(candidate_params), desc='RF tuning'), start=1):
        if args.selection_mode == 'train_cv_5fold':
            metrics = evaluate_candidate_with_train_cv(
                params,
                train_df,
                y_train,
                seeds=eval_seeds,
                rf_jobs=args.rf_jobs,
                scale_features=scale_features,
                cv_n_splits=5,
                cv_seed=args.search_seed,
            )
        else:
            metrics = evaluate_candidate(params, X_train, y_train, X_valid, y_valid, eval_seeds, args.rf_jobs)
        record = {
            'candidate_index': index,
            **params,
            **metrics,
        }
        results.append(record)

        if best_record is None or (
            record[selection_mean_key] > best_record[selection_mean_key]
            or (
                record[selection_mean_key] == best_record[selection_mean_key]
                and record[selection_std_key] < best_record[selection_std_key]
            )
        ):
            best_record = record
            selection_label = 'cv' if args.selection_mode == 'train_cv_5fold' else 'valid'
            print(
                f"New best candidate #{index}: "
                f"{selection_label} {args.selection_metric}={record[selection_mean_key]:.6f} +/- {record[selection_std_key]:.6f}"
            )

    output_dir = Path(args.output_dir) / args.subtask
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results).sort_values(
        by=[selection_mean_key, selection_std_key, 'mean_valid_roc_auc'],
        ascending=[False, True, False],
    )
    results_csv = output_dir / f'{args.subtask}_{args.feature_backend}_rf_search_results.csv'
    results_df.to_csv(results_csv, index=False)

    best_params = extract_best_params(best_record)
    final_model_metrics = None
    final_eval_seed = None
    if args.selection_mode == 'train_cv_5fold':
        final_eval_seed = eval_seeds[0]
        final_model = RandomForestClassifier(
            **best_params,
            random_state=final_eval_seed,
            n_jobs=args.rf_jobs,
        )
        final_model.fit(X_train, y_train)
        train_proba = final_model.predict_proba(X_train)[:, 1]
        valid_proba = final_model.predict_proba(X_valid)[:, 1]
        train_pred = final_model.predict(X_train)
        valid_pred = final_model.predict(X_valid)
        final_model_metrics = {
            'train': {
                'roc_auc': compute_roc_auc(y_train, train_proba),
                'macro_f1': compute_macro_f1(y_train, train_pred),
            },
            'valid': {
                'roc_auc': compute_roc_auc(y_valid, valid_proba),
                'macro_f1': compute_macro_f1(y_valid, valid_pred),
            },
        }

    best_output = {
        'subtask': args.subtask,
        'feature_backend': args.feature_backend,
        'selection_metric': args.selection_metric,
        'selection_mode': args.selection_mode,
        'search_seed': args.search_seed,
        'eval_seeds': eval_seeds,
        'best_params': best_params,
        'best_metrics': {
            'mean_train_roc_auc': best_record['mean_train_roc_auc'],
            'std_train_roc_auc': best_record['std_train_roc_auc'],
            'mean_valid_roc_auc': best_record['mean_valid_roc_auc'],
            'std_valid_roc_auc': best_record['std_valid_roc_auc'],
            'mean_train_macro_f1': best_record['mean_train_macro_f1'],
            'std_train_macro_f1': best_record['std_train_macro_f1'],
            'mean_valid_macro_f1': best_record['mean_valid_macro_f1'],
            'std_valid_macro_f1': best_record['std_valid_macro_f1'],
        },
        'num_surviving_features': len(surviving_columns),
        'surviving_feature_names': surviving_columns,
    }
    if args.selection_mode == 'train_cv_5fold':
        best_output['cv_num_folds'] = 5
    if final_model_metrics is not None:
        best_output['final_eval_seed'] = final_eval_seed
        best_output['final_model_metrics'] = final_model_metrics
    best_json = output_dir / f'{args.subtask}_{args.feature_backend}_best_params.json'
    with best_json.open('w', encoding='utf-8') as f:
        json.dump(best_output, f, indent=2, ensure_ascii=False)

    print(f'Saved search results to {results_csv}')
    print(f'Saved best-parameter summary to {best_json}')
    print('Best params:')
    print(json.dumps(best_params, indent=2, ensure_ascii=False))
    selection_label = 'cv' if args.selection_mode == 'train_cv_5fold' else 'valid'
    print(
        f"Best {selection_label} {args.selection_metric}: {best_record[selection_mean_key]:.6f} "
        f"+/- {best_record[selection_std_key]:.6f}"
    )
    if final_model_metrics is not None:
        print(
            f"Final valid {args.selection_metric}: "
            f"{final_model_metrics['valid'][args.selection_metric]:.6f}"
        )


if __name__ == '__main__':
    raise SystemExit(main())
