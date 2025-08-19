#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSG window extraction & labeling pipeline
---------------------------------------

Converts EDFs + event TXT files into per-window NumPy arrays for ML.

Expected input layout:
  <events_dir>/SUBJECT_ID epoch and event list.txt
  <edf_dir>/SUBJECT_ID_New.edf

Outputs (per subject/window):
  <out_dir>/<SUBJECT_ID>_win_XXX_<pos|neg>.npy         # shape: (n_channels, n_times)
  <out_dir>/<SUBJECT_ID>_win_XXX_<pos|neg>_times.npy  # shape: (n_times,)
  <out_dir>/channel_names.npy                         # first subject's channel order

Example:
  python psg_window_pipeline.py \
    --events-dir /path/to/Events_data \
    --edf-dir /path/to/EDF_data \
    --out-dir /path/to/Npy_data \
    --window-length 10 --delta 5 --beta 20 --gamma 30 \
    --channels M1,M2,C3,C4
"""
from __future__ import annotations

import os
import sys
import glob
import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import mne

from tqdm import tqdm
# -----------------------------
# Utilities & configuration
# -----------------------------

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep MNE quiet unless debugging
    mne.set_log_level("WARNING" if not verbose else "INFO")


@dataclass
class PipelineConfig:
    events_dir: str
    edf_dir: str
    out_dir: str
    window_length: float = 10.0
    delta: float = 5.0
    beta: float = 20.0
    gamma: float = 30.0
    min_gap: float = 60.0
    channels: Optional[List[str]] = None
    balance_negatives: bool = True
    max_subjects: Optional[int] = None
    preload_edf: bool = False


# -----------------------------
# Event parsing & windowing
# -----------------------------

def read_events(txt_path: str, edf_start: datetime, start_line: int = 1) -> pd.DataFrame:
    """Parse an event TXT and compute seconds since EDF start.

    Assumes the file has a header line with column names at ``start_line`` and
    that the recording start timestamp is at line index 4, column 2 (tab-delimited).
    """
    with open(txt_path, 'r', encoding='latin-1') as f:
        lines = [ln.rstrip('\n') for ln in f]

    if len(lines) <= start_line:
        raise ValueError(f"Event file seems too short: {txt_path}")

    # Build DataFrame from data lines
    data = [ln.split('\t') for ln in lines[start_line:] if ln.strip()]
    df = pd.DataFrame(data)

    # First data row mirrors header; set columns and drop it
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)

    # Start Time is hh:mm:ss AM/PM; combine with EDF date to get absolute
    df['time_only'] = pd.to_datetime(df['Start Time'], format="%I:%M:%S %p", errors='coerce').dt.time
    df['Start Time'] = df['time_only'].apply(lambda t: datetime.combine(edf_start.date(), t))
    # Handle wrap past midnight
    df.loc[df['Start Time'] < edf_start, 'Start Time'] += timedelta(days=1)

    df['seconds_since_start'] = (df['Start Time'] - edf_start).dt.total_seconds()
    return df


def make_sample_windows(
    events_df: pd.DataFrame,
    window_length: float = 10.0,
    delta: float = 0.0,
    beta: float = 20.0,
    gamma: float = 30.0,
) -> pd.DataFrame:
    """Create candidate windows around hypopnea events.

    For each hypopnea event time ``t`` we form up to two windows:
      - negative: [t - delta - gamma - L, t - delta - gamma)
      - positive (if delta >= 0): [t - delta - L, t - delta)

    We enforce:
      * no overlap with previously accepted windows
      * windows must be within the recording (start >= 0)
      * event-level and window-level spacing via ``beta``
    """
    hypnos = events_df.loc[
        events_df['Event'].str.contains('Hypopnea', na=False),
        ['Start Time', 'seconds_since_start']
    ].sort_values('seconds_since_start').reset_index(drop=True)

    logging.debug("Found %d hypopnea events for window creation", len(hypnos))

    accepted: List[Tuple[float, float]] = []
    samples = []
    skip_event = -np.inf

    for i, (t, event_start_time) in enumerate(zip(hypnos['seconds_since_start'], hypnos['Start Time'])):
        if t < skip_event:
            continue

        pos_end = t - delta
        pos_start = pos_end - window_length
        neg_end = pos_start - gamma
        neg_start = neg_end - window_length

        logging.debug("Event %d at t=%.2f: pos=[%.2f, %.2f], neg=[%.2f, %.2f]", 
                     i, t, pos_start, pos_end, neg_start, neg_end)

        candidates = [('neg', neg_start, neg_end, event_start_time)]
        if delta >= 0:
            candidates.append(('pos', pos_start, pos_end, event_start_time))

        for label, st, en, estart in candidates:
            if en <= st:
                logging.debug("Skipping %s window (invalid range): [%.2f, %.2f]", label, st, en)
                continue
            if st < 0:
                logging.debug("Skipping %s window (starts before recording): [%.2f, %.2f]", label, st, en)
                continue
            # Overlap check
            if any(not (en <= a_st or st >= a_en) for a_st, a_en in accepted):
                logging.debug("Skipping %s window (overlaps existing): [%.2f, %.2f]", label, st, en)
                continue

            logging.debug("Accepting %s window: [%.2f, %.2f] (duration=%.2f)", label, st, en, en - st)
            samples.append({
                'event_time': t,
                'label': label,
                'start': st,
                'end': en,
                'event_start': estart,
            })
            accepted.append((st, en))

        # event-level beta spacing
        skip_event = t + beta

    return pd.DataFrame(samples)


def add_extra_negative_windows(
    existing_samples: pd.DataFrame,
    events_df: pd.DataFrame,
    recording_duration: float,
    window_length: float = 10.0,
    beta: float = 20.0,
    delta: float = 0.0,
    target_extra_negatives: Optional[int] = None,
    min_gap: float = 60.0,
) -> pd.DataFrame:
    """Add additional background windows avoiding forbidden zones around hypopnea.

    """
    if existing_samples.empty:
        logging.info("No existing samples; skipping extra negatives.")
        return existing_samples

    hypopnea_times = (events_df
                      .query("Event.str.contains('Hypopnea', na=False)")
                      .seconds_since_start
                      .sort_values()
                      .to_numpy())

    logging.debug("Avoiding %d hypopnea events.", len(hypopnea_times))

    existing_windows = existing_samples[['start', 'end']].sort_values('start').values

    forbidden_zones = [(t - delta, t + beta) for t in hypopnea_times]

    def is_window_valid(wst: float, wen: float) -> bool:
        for zst, zen in forbidden_zones:
            if not (wen <= zst or wst >= zen):
                return False
        return True

    available_periods: List[Tuple[float, float]] = []
    if len(existing_windows) > 0:
        first_start = existing_windows[0][0]
        if first_start > window_length + min_gap:
            available_periods.append((0.0, first_start - min_gap))

    for i in range(len(existing_windows) - 1):
        gap_start = existing_windows[i][1] + min_gap
        gap_end = existing_windows[i + 1][0] - min_gap
        if gap_end - gap_start >= window_length:
            available_periods.append((gap_start, gap_end))

    if len(existing_windows) > 0:
        last_end = existing_windows[-1][1]
        if recording_duration - last_end > window_length + min_gap:
            available_periods.append((last_end + min_gap, recording_duration))

    logging.debug("Found %d available periods for extra negatives.", len(available_periods))

    extra_negatives = []
    total_added = 0

    for p_st, p_en in available_periods:
        period_len = p_en - p_st
        max_in_period = int((period_len - window_length) / (window_length + beta)) + 1
        if max_in_period <= 0:
            continue

        if target_extra_negatives is not None:
            remaining = max(0, target_extra_negatives - total_added)
            if remaining == 0:
                break
            to_add = min(max_in_period, remaining)
        else:
            to_add = max_in_period

        usable_length = period_len - window_length
        positions = [p_st + i * (usable_length / (to_add - 1)) for i in range(to_add)] if to_add > 1 else [p_st + usable_length / 2]

        added_here = 0
        for wst in positions:
            wen = wst + window_length
            if wen > p_en:
                continue
            if is_window_valid(wst, wen):
                logging.debug("Adding extra negative window: [%.2f, %.2f] (duration=%.2f)", wst, wen, wen - wst)
                extra_negatives.append({
                    'event_time': np.nan,
                    'label': 'neg',
                    'start': wst,
                    'end': wen,
                    'event_start': np.nan,
                })
                total_added += 1
                added_here += 1
                if target_extra_negatives is not None and total_added >= target_extra_negatives:
                    break
        logging.debug("Period %.1f-%.1fs: added %d windows", p_st, p_en, added_here)

    if extra_negatives:
        combined = pd.concat([existing_samples, pd.DataFrame(extra_negatives)], ignore_index=True)
        return combined.sort_values('start').reset_index(drop=True)
    return existing_samples


# -----------------------------
# EDF extraction
# -----------------------------

def extract_from_edf(
    edf_path: str,
    windows_df: pd.DataFrame,
    channels: Optional[List[str]] = None,
    preload: bool = False,
) -> Tuple[list, float, List[str], datetime]:
    """Slice raw EDF for each window.

    Returns (samples, duration_seconds, channel_names, edf_start)
    where samples is a list of dicts: {'data', 'times', 'label', 'ch_names'}
    """
    logging.info("=" * 60)
    logging.info("EXTRACTING FROM EDF: %s", os.path.basename(edf_path))
    
    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose=False)
    raw.resample(256, npad='auto')  # Resample to 256 Hz if not already

    # Channel selection
    if channels:
        keep = [ch for ch in channels if ch in raw.ch_names]
        if not keep:
            logging.warning("None of the requested channels %s are present in %s; using all channels.", channels, os.path.basename(edf_path))
        else:
            raw.pick_channels(keep, ordered=True)

    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    total_samples = raw.n_times
    edf_start = raw.info['meas_date']
    if edf_start is None:
        # Fallback to 1970 epoch if m\issing
        edf_start = datetime.fromtimestamp(0)
    else:
        edf_start = edf_start.replace(tzinfo=None)

    duration_sec = total_samples / float(sfreq)

    logging.info(
        "EDF %s | channels=%s | samples=%d | sfreq=%.2f Hz | duration=%.2f sec | start=%s",
        os.path.basename(edf_path), ch_names, total_samples, sfreq, duration_sec, edf_start,
    )

    # DEBUG: Check if we have any suspicious sampling frequencies
    if abs(sfreq - 256) > 1:  # Most PSG files are 256 Hz
        logging.warning("  UNUSUAL SAMPLING FREQUENCY: %.2f Hz (expected ~256 Hz)", sfreq)

    out = []
    problematic_windows = []
    
    for i, (_, row) in enumerate(windows_df.iterrows()):
        start_time = float(row.start)
        end_time = float(row.end)
        expected_duration = end_time - start_time
        
        logging.debug("Window %03d [%s]: time range [%.2f, %.2f] sec (duration=%.2f sec)", 
                     i, row.label, start_time, end_time, expected_duration)
        
        # Convert time to sample indices
        start_idx = raw.time_as_index(start_time)[0]
        end_idx = raw.time_as_index(end_time)[0]
        
        expected_samples = int(expected_duration * sfreq)
        actual_samples = end_idx - start_idx
        
        logging.debug("  → Sample indices: [%d, %d] (span=%d samples)", start_idx, end_idx, actual_samples)
        logging.debug("  → Expected samples: %d (%.2f sec × %.2f Hz)", expected_samples, expected_duration, sfreq)
        
        # Check for discrepancies
        sample_discrepancy = abs(actual_samples - expected_samples)
        if sample_discrepancy > sfreq * 0.1:  # More than 0.1 second difference
            logging.error(" SAMPLE COUNT MISMATCH in window %03d [%s]:", i, row.label)
            logging.error("   Expected: %d samples (%.2f sec)", expected_samples, expected_duration)
            logging.error("   Actual:   %d samples (%.2f sec)", actual_samples, actual_samples / sfreq)
            logging.error("   File: %s", os.path.basename(edf_path))
            problematic_windows.append({
                'window_idx': i,
                'label': row.label,
                'expected_samples': expected_samples,
                'actual_samples': actual_samples,
                'expected_duration': expected_duration,
                'actual_duration': actual_samples / sfreq
            })

        if end_idx > total_samples:
            logging.warning("Window %03d [%s] exceeds EDF length (end_idx=%d > %d); skipping.", i, row.label, end_idx, total_samples)
            continue

        # Extract the data
        data, times = raw[:, start_idx:end_idx]
        
        if data.size == 0:
            logging.warning("Window %03d [%s] produced empty slice; skipping.", i, row.label)
            continue

        # Verify extracted data dimensions
        n_channels, n_timepoints = data.shape
        actual_duration_from_data = n_timepoints / sfreq
        
        logging.debug("  → Extracted data: (%d channels, %d timepoints) = %.3f seconds", 
                     n_channels, n_timepoints, actual_duration_from_data)
        
        # Flag windows that are significantly different from expected 10 seconds
        if abs(actual_duration_from_data - 10.0) > 0.5:  # More than 0.5 sec off from 10 sec
            logging.error(" DURATION ANOMALY in window %03d [%s]: %.2f seconds (expected ~10.0)", 
                         i, row.label, actual_duration_from_data)
            logging.error("   Data shape: %s, Sampling rate: %.2f Hz", data.shape, sfreq)
            logging.error("   File: %s", os.path.basename(edf_path))

        if end_idx - start_idx != 2560:
            print("WARNING: Window %03d [%s] has unexpected sample count: %d (expected 2560 for 10 sec @ 256 Hz)" % (i, row.label, end_idx - start_idx), edf_path)

        out.append({
            'data': data,           # (n_channels, n_times)
            'times': times,         # (n_times,)
            'label': row.label,
            'ch_names': ch_names,
        })

    # Summary of problematic windows for this file
    if problematic_windows:
        logging.error(" SUMMARY for %s: Found %d problematic windows", 
                     os.path.basename(edf_path), len(problematic_windows))
        for pw in problematic_windows:
            logging.error("   Window %03d [%s]: %.2f sec (expected %.2f)", 
                         pw['window_idx'], pw['label'], pw['actual_duration'], pw['expected_duration'])

    logging.info("Successfully extracted %d/%d windows from %s", len(out), len(windows_df), os.path.basename(edf_path))
    logging.info("=" * 60)
    
    return out, duration_sec, ch_names, edf_start


# -----------------------------
# Main pipeline
# -----------------------------

def process_subject(
    subject_id: str,
    cfg: PipelineConfig,
    channels_saved: bool,
) -> Tuple[int, int, bool]:
    """Process a single subject. Returns (n_pos, n_neg, channels_saved_now)."""
    txt_path = os.path.join(cfg.events_dir, f"{subject_id} epoch and event list_new.txt")
    edf_path = os.path.join(cfg.edf_dir, f"{subject_id}_New.edf")

    logging.info("\n" + " " + "="*70)
    logging.info(" PROCESSING SUBJECT: %s", subject_id)
    logging.info(" " + "="*70)
    logging.info("Events file: %s", txt_path)
    logging.info(" EDF file: %s", edf_path)

    # Check if both files exist
    if not os.path.exists(txt_path):
        logging.warning("Events file missing for %s: %s", subject_id, txt_path)
        return 0, 0, channels_saved
    
    if not os.path.exists(edf_path):
        logging.warning("EDF missing for %s: %s", subject_id, edf_path)
        return 0, 0, channels_saved

    # Need EDF start time to anchor events
    raw_hdr = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    raw_hdr.resample(256, npad='auto')  # Resample to 256 Hz if not already
    edf_start = raw_hdr.info['meas_date']
    if edf_start is None:
        edf_start = datetime.fromtimestamp(0)
    else:
        edf_start = edf_start.replace(tzinfo=None)

    logging.info(" EDF start time: %s", edf_start)
    logging.info(" Configured window length: %.2f seconds", cfg.window_length)

    events = read_events(txt_path, edf_start)
    logging.info(" Read %d events from %s", len(events), os.path.basename(txt_path))
    
    windows = make_sample_windows(
        events,
        window_length=cfg.window_length,
        delta=cfg.delta,
        beta=cfg.beta,
        gamma=cfg.gamma,
    )

    pos_count = int((windows['label'] == 'pos').sum())
    neg_count = int((windows['label'] == 'neg').sum())
    logging.info(" Initial window counts: %d positive, %d negative", pos_count, neg_count)

    # Log all windows for debugging
    for i, (_, row) in enumerate(windows.iterrows()):
        duration = row.end - row.start
        logging.debug("  Window %03d [%s]: [%.2f, %.2f] → duration=%.2f sec", 
                     i, row.label, row.start, row.end, duration)

    # Extract once to get duration - using empty dataframe to avoid processing all windows twice
    sample_window = windows.head(1) if not windows.empty else pd.DataFrame()
    if not sample_window.empty:
        _, duration_sec_tmp, _, _ = extract_from_edf(edf_path, sample_window, channels=cfg.channels, preload=cfg.preload_edf)
        logging.info(" Total recording duration: %.2f seconds", duration_sec_tmp)
    else:
        logging.warning("No windows to process for %s", subject_id)
        return 0, 0, channels_saved

    if cfg.balance_negatives and pos_count > neg_count:
        target_extra = pos_count - neg_count
        logging.info("  Balancing: need %d extra negative windows", target_extra)
        windows = add_extra_negative_windows(
            windows,
            events,
            recording_duration=duration_sec_tmp,
            window_length=cfg.window_length,
            beta=cfg.beta,
            delta=cfg.delta,
            target_extra_negatives=target_extra,
            min_gap=cfg.min_gap,
        )
        pos_count = int((windows['label'] == 'pos').sum())
        neg_count = int((windows['label'] == 'neg').sum())
        logging.info("  After balancing: %d positive, %d negative", pos_count, neg_count)

    # Now extract all windows
    samples, duration_sec, ch_names, _ = extract_from_edf(
        edf_path, windows, channels=cfg.channels, preload=cfg.preload_edf
    )

    os.makedirs(cfg.out_dir, exist_ok=True)

    # Save channel names once
    if (not channels_saved) and samples:
        np.save(os.path.join(cfg.out_dir, "channel_names.npy"), np.array(ch_names))
        channels_saved = True
        # logging.info(" Saved channel names: %s", ch_names)

    # Persist windows with additional validation
    pos_written = 0
    neg_written = 0
    for i, seg in enumerate(samples):
        label = str(seg['label'])
        data = seg['data']
        times = seg['times']
        
        # Additional validation on saved data
        n_channels, n_samples = data.shape
        duration_from_samples = len(times)
        
        logging.debug(" Saving window %03d [%s]: shape=%s, times_len=%d", 
                     i, label, data.shape, len(times))
        
        # Check if this window has unexpected duration
        if n_samples == 5120:  # This is your 20-second problem!
            logging.error("   FOUND 20-SECOND WINDOW! Window %03d [%s] in subject %s", i, label, subject_id)
            logging.error("   Data shape: %s", data.shape)
            logging.error("   Times array length: %d", len(times))
            logging.error("   Expected ~2560 samples for 10 sec @ 256 Hz")
            logging.error("   File: %s", edf_path)
        
        if label == 'pos':
            pos_written += 1
        else:
            neg_written += 1
            
        fn_base = os.path.join(cfg.out_dir, f"{subject_id}_win_{i:03d}_{label}")
        np.save(f"{fn_base}.npy", seg['data'])
        np.save(f"{fn_base}_times.npy", seg['times'])

    logging.info("✅ Saved %d windows for %s (recording duration=%.1fs)", len(samples), subject_id, duration_sec)
    logging.info("   Final counts: %d positive, %d negative", pos_written, neg_written)
    
    return pos_written, neg_written, channels_saved


def discover_subject_ids(events_dir: str, edf_dir: str) -> List[str]:
    """Discover subject IDs by looking for '*epoch and event list.txt' files and matching EDF files."""
    # Find all event files
    txt_paths = glob.glob(os.path.join(events_dir, "*epoch and event list_new.txt"))
    event_ids = [os.path.basename(p).replace(" epoch and event list_new.txt", "") for p in txt_paths]
    
    # Find all EDF files
    edf_paths = glob.glob(os.path.join(edf_dir, "*_New.edf"))
    edf_ids = [os.path.basename(p).replace("_New.edf", "") for p in edf_paths]
    
    # Find intersection (subjects that have both files)
    common_ids = list(set(event_ids) & set(edf_ids))
    common_ids.sort()
    
    logging.info("📊 Discovery results:")
    logging.info("   Events files found: %d subjects", len(event_ids))
    logging.info("   EDF files found: %d subjects", len(edf_ids))
    logging.info("   Common subjects: %d subjects", len(common_ids))
    
    if len(common_ids) < len(event_ids):
        missing_edf = set(event_ids) - set(edf_ids)
        if missing_edf:
            logging.warning("   Subjects with events but no EDF: %s", sorted(missing_edf))
    
    if len(common_ids) < len(edf_ids):
        missing_events = set(edf_ids) - set(event_ids)
        if missing_events:
            logging.warning("   Subjects with EDF but no events: %s", sorted(missing_events))
    
    return common_ids


def parse_args(argv: Optional[List[str]] = None) -> PipelineConfig:
    ap = argparse.ArgumentParser(description="PSG EDF → windowed NumPy exporter")
    ap.add_argument("--events-dir", required=True, help="Folder containing event TXT files")
    ap.add_argument("--edf-dir", required=True, help="Folder containing EDF files")
    ap.add_argument("--out-dir", required=True, help="Output folder for .npy windows")
    ap.add_argument("--window-length", type=float, default=10.0, help="Window length in seconds")
    ap.add_argument("--delta", type=float, default=5.0, help="Gap before event for pos window end")
    ap.add_argument("--beta", type=float, default=20.0, help="Post-event spacing and forbidden zone size")
    ap.add_argument("--gamma", type=float, default=30.0, help="Neg window separation before pos window")
    ap.add_argument("--min-gap", type=float, default=60.0, help="Min gap from existing windows for extra negatives")
    ap.add_argument("--channels", type=str, default="M1,M2,C3,C4,RIBCAGE,ABDOMEN,SaO2,NASAL PRES.", help="Comma-separated channel list to keep; use 'ALL' for all channels")
    ap.add_argument("--no-balance", action="store_true", help="Do not add extra negative windows to balance positives")
    ap.add_argument("--max-subjects", type=int, default=None, help="Limit number of subjects for a quick run")
    ap.add_argument("--preload-edf", action="store_true", help="Preload EDFs into memory (faster but uses more RAM)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = ap.parse_args(argv)

    channels = None if args.channels.strip().upper() == 'ALL' else [s.strip() for s in args.channels.split(',') if s.strip()]

    return PipelineConfig(
        events_dir=args.events_dir,
        edf_dir=args.edf_dir,
        out_dir=args.out_dir,
        window_length=args.window_length,
        delta=args.delta,
        beta=args.beta,
        gamma=args.gamma,
        min_gap=args.min_gap,
        channels=channels,
        balance_negatives=not args.no_balance,
        max_subjects=args.max_subjects,
        preload_edf=False,
    )


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    # Determine if we should be verbose from argv
    verbose = argv and '-v' in argv or '--verbose' in argv if argv else '-v' in sys.argv or '--verbose' in sys.argv
    setup_logging(verbose=verbose)

    os.makedirs(cfg.out_dir, exist_ok=True)

    subject_ids = discover_subject_ids(cfg.events_dir, cfg.edf_dir)
    if not subject_ids:
        logging.error("No subjects found with both events and EDF files")
        logging.error("  Events dir: %s", cfg.events_dir)
        logging.error("  EDF dir: %s", cfg.edf_dir)
        return 2

    if cfg.max_subjects:
        subject_ids = subject_ids[: cfg.max_subjects]

    logging.info("Discovered %d subject(s). Output → %s", len(subject_ids), cfg.out_dir)
    logging.info("Configuration: window_length=%.2f, delta=%.2f, beta=%.2f, gamma=%.2f", 
                cfg.window_length, cfg.delta, cfg.beta, cfg.gamma)

    total_pos = 0
    total_neg = 0
    channels_saved = False
    problematic_subjects = []

    for subject_idx, sid in tqdm(enumerate(subject_ids[20:]), total=457):
        logging.info("\nProcessing subject %d/%d: %s", subject_idx + 1, len(subject_ids), sid)
        try:
            p, n, channels_saved = process_subject(sid, cfg, channels_saved)
            total_pos += p
            total_neg += n
            
            # Check if this subject had any 20-second windows by looking for the error message in logs
            # (This is a simple heuristic - in production you might want more sophisticated tracking)
            
        except Exception as e:
            logging.exception("Failed to process %s: %s", sid, e)
            problematic_subjects.append(sid)

    logging.info("\n" + "🎉 " + "="*60)
    logging.info("PROCESSING COMPLETE!")
    logging.info("🎉 " + "="*60)
    logging.info("Total windows: Positive=%d, Negative=%d", total_pos, total_neg)
    
    if problematic_subjects:
        logging.warning("Failed subjects (%d): %s", len(problematic_subjects), problematic_subjects)
    
    logging.info("Output directory: %s", cfg.out_dir)
    logging.info("Done!")
    
    return 0


if __name__ == "__main__":
    main()