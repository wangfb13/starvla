#!/usr/bin/env python3
"""Held-out, expert-observation evaluation; this is not a closed-loop rollout."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from offline_eval_starvla_event_labels import (
    ACTION, OBS_STATE, PreTrainedConfig, build_dataset, current_state_from_history,
    make_policy, make_pre_post_processors, parse_ints, plot_samples, set_seed,
    stable_close_index, summarize_prediction, to_int, to_keyframe_relative_actions,
    to_numpy,
)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ['checkpoint', 'base-vlm', 'dataset-root', 'sidecar', 'relative-stats', 'output-dir']:
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--num-seeds', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261025)
    p.add_argument('--close-threshold', type=float, default=0.55)
    p.add_argument('--stable-steps', type=int, default=4)
    p.add_argument('--auto-anchor-offsets', default='50,32,10')
    p.add_argument('--device', default='cuda')
    p.add_argument('--episodes', default=None, help='Optional comma-separated subset; default all.')
    return p.parse_args()


def distribution(values):
    a = np.asarray([x for x in values if x is not None], dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'n': 0}
    return {key: float(value) for key, value in {
        'mean': a.mean(), 'median': np.median(a), 'p90': np.quantile(a, .90),
        'p95': np.quantile(a, .95), 'min': a.min(), 'max': a.max(),
    }.items()} | {'n': int(len(a))}


def aggregate(rows):
    events = [r for r in rows if r['label_contains_event'] and r['label_contains_stable_close']]
    no_close = [r for r in rows if not r['label_contains_stable_close'] and not r['label_already_closed_at_window_start']]
    metrics = ['xyz_mae_mm', 'xyz_error_at_label_event_mm', 'rotation_error_at_label_event_deg',
               'event_to_event_close_position_error_mm', 'prediction_close_frame_error']
    return {
        'all_prediction_count': len(rows),
        'event_prediction_count': len(events),
        'predicted_close_count': sum(r['prediction_contains_stable_close'] for r in events),
        'missed_close_count': sum(not r['prediction_contains_stable_close'] for r in events),
        'within_1_frame_count': sum(r['prediction_close_frame_error'] is not None and abs(r['prediction_close_frame_error']) <= 1 for r in events),
        'within_3_frames_count': sum(r['prediction_close_frame_error'] is not None and abs(r['prediction_close_frame_error']) <= 3 for r in events),
        'no_close_label_prediction_count': len(no_close),
        'no_close_label_false_positive_count': sum(r['prediction_contains_stable_close'] for r in no_close),
        'metrics': {k: distribution([r[k] for r in events]) for k in metrics},
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    args = args_parser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    labels = pd.read_parquet(args.sidecar)
    labels = labels.sort_values(['episode_index', 'frame_index'])
    available_episodes = sorted(int(x) for x in labels.episode_index.unique())
    episodes = parse_ints(args.episodes) if args.episodes else available_episodes
    if not set(episodes).issubset(available_episodes):
        raise ValueError('Requested episode is absent')
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    cfg.pretrained_path = str(args.checkpoint)
    cfg.base_vlm = str(args.base_vlm)
    cfg.device, cfg.use_amp = args.device, False
    cfg.online_anchor_state_to_episode_start = False
    gap, chunk_size = int(cfg.action_gap_frames), int(cfg.chunk_size)
    assert (gap, chunk_size) == (6, 32)
    print(f'[heldout] episodes={len(episodes)}; building actual training dataset/processor', flush=True)
    dataset = build_dataset(args)
    policy = make_policy(cfg, ds_meta=dataset.meta, rename_map={})
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=str(args.checkpoint),
        preprocessor_overrides={
            'device_processor': {'device': args.device},
            'rename_observations_processor': {'rename_map': {}},
            'starvla_prepare_tactile_history': {
                'enabled': True, 'history_len': int(cfg.tactile_history_len),
                'force_indices': list(cfg.tactile_force_indices), 'output_key': cfg.tactile_history_key,
            },
        },
        postprocessor_overrides={'device_processor': {'device': 'cpu'}},
    )
    policy.eval()
    model = policy.starvla_model.qwen_vl_interface.model
    embedding_tied = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    if not embedding_tied:
        raise RuntimeError('Expected checkpoint lm_head and embedding to remain tied')
    rows, ep_summaries, skipped, cross_anchors = [], [], [], []
    max_roundtrip, max_mapping = 0., 0.
    offsets = parse_ints(args.auto_anchor_offsets)
    episodes_dir = args.output_dir / 'episodes'
    episodes_dir.mkdir(exist_ok=True)
    print('[heldout] model ready; label action key removed from inference', flush=True)
    with (args.output_dir / 'prediction_metrics.jsonl').open('w') as row_file:
        for episode in episodes:
            ep = labels[labels.episode_index == episode].reset_index(drop=True)
            actions = np.stack(ep.action.to_numpy()).astype(np.float32)
            close_index = stable_close_index(actions[:, 9], args.close_threshold, args.stable_steps, require_open_before=True)
            if close_index is None:
                skipped.append({'episode': episode, 'reason': 'no_open_to_stable_close_transition'})
                print(f'[heldout] episode={episode} skipped: no label close', flush=True)
                continue
            event_frame = int(ep.iloc[close_index].frame_index)
            indexed = ep.set_index('frame_index')
            samples, episode_rows, saved, sample_checks = [], [], {}, []
            for offset in offsets:
                current = event_frame - offset
                requested = current + gap + np.arange(chunk_size)
                if current not in indexed.index or not set(requested).issubset(indexed.index):
                    skipped.append({'episode': episode, 'current_frame': current, 'offset': offset, 'reason': 'missing_or_padded_frame_window'})
                    continue
                future = indexed.loc[requested]
                if 'action_valid_mask' in future and not np.all(np.stack(future.action_valid_mask.to_numpy()) > .5):
                    skipped.append({'episode': episode, 'current_frame': current, 'offset': offset, 'reason': 'invalid_action_dimensions'})
                    continue
                dataset_index = int(indexed.loc[current, 'index'])
                item = dataset[dataset_index]
                assert to_int(item['episode_index']) == episode and to_int(item['frame_index']) == current
                raw_batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v] for k, v in item.items()}
                raw_target = raw_batch[ACTION].clone()
                expected = np.stack(future.action.to_numpy()).astype(np.float32)
                mapping = float(np.max(np.abs(to_numpy(raw_target[0]) - expected)))
                if mapping > 2e-6:
                    raise RuntimeError(f'Original frame mapping failed: ep={episode} t={current} error={mapping}')
                state = current_state_from_history(raw_batch[OBS_STATE].clone())
                policy.reset()
                preprocessor.reset()
                batch = preprocessor(raw_batch)
                target_model = batch[ACTION].detach().clone()
                inference_batch = {k: v for k, v in batch.items() if k != ACTION}
                assert ACTION not in inference_batch
                target_t = postprocessor(target_model.clone())
                roundtrip = float(torch.max(torch.abs(target_t.cpu() - raw_target.cpu())).item())
                if roundtrip > 2e-5:
                    raise RuntimeError(f'Clean-label normalization/decode failed: ep={episode} t={current} error={roundtrip}')
                max_mapping, max_roundtrip = max(max_mapping, mapping), max(max_roundtrip, roundtrip)
                target = to_numpy(target_t[0])
                target_local = to_numpy(to_keyframe_relative_actions(target_t, state, cfg.relative_action_keyframes, cfg.relative_exclude_keys)[0])
                predictions, local_predictions, model_predictions = [], [], []
                for seed_i in range(args.num_seeds):
                    actual_seed = args.seed + seed_i
                    set_seed(actual_seed)
                    with torch.inference_mode():
                        prediction_model = policy.predict_action_chunk(inference_batch)
                        prediction_t = postprocessor(prediction_model.clone())
                    prediction = to_numpy(prediction_t[0])
                    prediction_local = to_numpy(to_keyframe_relative_actions(prediction_t, state, cfg.relative_action_keyframes, cfg.relative_exclude_keys)[0])
                    row = summarize_prediction(
                        pred_absolute=prediction, target_absolute=target, current_frame=current,
                        gap=gap, label_event_frame=event_frame, threshold=args.close_threshold, stable_steps=args.stable_steps,
                    )
                    row.update(episode_index=episode, seed=actual_seed, anchor_offset=offset,
                               inference_action_key_present=False, clean_target_roundtrip_max_abs=roundtrip,
                               original_frame_mapping_max_abs=mapping)
                    row['prediction_closed_at_window_start'] = row['prediction_stable_close_chunk_index'] == 0
                    row['prediction_close_transition_observed'] = row['prediction_stable_close_chunk_index'] is not None and row['prediction_stable_close_chunk_index'] > 0
                    row['close_time_interpretation'] = ('upper_bound_transition_no_later_than_first_predicted_frame'
                                                        if row['prediction_closed_at_window_start'] else 'observed_transition' if row['prediction_close_transition_observed'] else 'no_stable_close_in_window')
                    row_file.write(json.dumps(row) + '\n')
                    rows.append(row)
                    episode_rows.append(row)
                    predictions.append(prediction)
                    local_predictions.append(prediction_local)
                    model_predictions.append(to_numpy(prediction_model[0]))
                event_k = event_frame - current - gap
                contains = 0 <= event_k < chunk_size
                predictions, local_predictions = np.stack(predictions), np.stack(local_predictions)
                key = f'frame_{current:04d}'
                saved.update({key + '_target_absolute': target, key + '_target_local': target_local,
                              key + '_clean_target_model': to_numpy(target_model[0]),
                              key + '_pred_absolute': predictions, key + '_pred_local': local_predictions,
                              key + '_pred_model': np.stack(model_predictions), key + '_current_state': to_numpy(state[0])})
                samples.append({'current_frame': current, 'dataset_index': dataset_index,
                                'label_event_frame': event_frame, 'label_event_chunk_index': event_k if contains else None,
                                'close_threshold': args.close_threshold, 'target_absolute': target,
                                'target_local': target_local, 'pred_absolute_all': predictions, 'pred_local_all': local_predictions})
                sample_checks.append({'current_frame': current, 'offset': offset, 'event_chunk_index': event_k if contains else None,
                                      'roundtrip_max_abs': roundtrip, 'original_frame_mapping_max_abs': mapping})
            event_samples = [s for s in samples if s['label_event_chunk_index'] is not None]
            if len(event_samples) == 2:
                a, b = event_samples
                ai, bi = a['label_event_chunk_index'], b['label_event_chunk_index']
                target_diff = float(np.linalg.norm(a['target_absolute'][ai, :3] - b['target_absolute'][bi, :3]) * 1000)
                if target_diff > .03:
                    raise RuntimeError(f'Shared event label mismatch: episode={episode} error_mm={target_diff}')
                for seed_i in range(args.num_seeds):
                    cross_anchors.append({'episode_index': episode, 'seed': args.seed + seed_i,
                                          'current_frames': [a['current_frame'], b['current_frame']],
                                          'event_frame': event_frame, 'label_cross_anchor_distance_mm': target_diff,
                                          'pred_cross_anchor_distance_mm': float(np.linalg.norm(a['pred_absolute_all'][seed_i, ai, :3] - b['pred_absolute_all'][seed_i, bi, :3]) * 1000)})
            if samples:
                np.savez_compressed(episodes_dir / f'episode_{episode:04d}.npz', **saved)
                ep_result = {'episode_index': episode, 'event_frame': event_frame, 'samples': sample_checks, 'aggregate': aggregate(episode_rows)}
                write_json(episodes_dir / f'episode_{episode:04d}.json', ep_result)
                ep_summaries.append(ep_result)
                print(f'[heldout] episode={episode} event={event_frame} samples={len(samples)} event_xyz={ep_result["aggregate"]["metrics"]["xyz_error_at_label_event_mm"]}', flush=True)
                # Keep a deterministic first-episode figure; select median/worst after all results exist.
                if len(ep_summaries) == 1:
                    plot_samples(samples, gap, args.output_dir / 'first_episode_prediction_vs_label.png')
            row_file.flush()
    summary = {
        'evaluation_type': 'heldout_expert_observations_not_closed_loop',
        'checkpoint': str(args.checkpoint), 'dataset': str(args.dataset_root), 'sidecar': str(args.sidecar),
        'training_relative_stats_for_dataset_metadata': str(args.relative_stats),
        'normalization_source': 'checkpoint saved preprocessor and postprocessor only',
        'model_embedding_and_lm_head_tied': embedding_tied,
        'model_task_override': cfg.task_override,
        'requested_episode_count': len(episodes), 'evaluated_episode_count': len(ep_summaries),
        'requested_episode_indices': episodes, 'skipped': skipped,
        'num_seeds': args.num_seeds, 'anchor_offsets': offsets,
        'close_definition': {'threshold': args.close_threshold, 'stable_steps': args.stable_steps, 'event': 'first stable close after an open label'},
        'chunk': {'size': chunk_size, 'gap': gap},
        'verification': {'action_removed_from_all_inference_batches': True,
                         'max_original_frame_mapping_abs': max_mapping, 'max_label_roundtrip_abs': max_roundtrip},
        'aggregate': aggregate(rows),
        'by_anchor_offset': {str(offset): aggregate([r for r in rows if r['anchor_offset'] == offset]) for offset in offsets},
        'shared_event_cross_anchor': {
            'count': len(cross_anchors),
            'label_distance_mm': distribution([r['label_cross_anchor_distance_mm'] for r in cross_anchors]),
            'prediction_distance_mm': distribution([r['pred_cross_anchor_distance_mm'] for r in cross_anchors]),
        },
        'episode_summaries': ep_summaries, 'elapsed_s': time.perf_counter() - started,
    }
    write_json(args.output_dir / 'summary.json', summary)
    write_json(args.output_dir / 'shared_event_cross_anchor.json', cross_anchors)
    write_json(args.output_dir / 'COMPLETE.json', {'completed': True, 'elapsed_s': summary['elapsed_s']})
    print('[heldout] COMPLETED ' + json.dumps(summary['aggregate']), flush=True)


if __name__ == '__main__':
    main()
