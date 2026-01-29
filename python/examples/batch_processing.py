#!/usr/bin/env python3
"""
Batch Processing Example

Demonstrates the performance benefits of plan reuse for batch processing.
"""

import numpy as np
import time
import spectrograms as sg


def generate_random_signal(sample_rate, duration):
    """Generate a random test signal."""
    n_samples = int(sample_rate * duration)
    return np.random.randn(n_samples).astype(np.float64)


def main():
    print("=" * 60)
    print("Batch Processing Example")
    print("=" * 60)

    # Configuration
    sample_rate = 16000
    duration = 1.0
    n_signals = 50  # Number of signals to process

    print(f"\nConfiguration:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration} s")
    print(f"  Number of signals: {n_signals}")
    print(f"  Samples per signal: {int(sample_rate * duration)}")

    # Generate test signals
    print(f"\nGenerating {n_signals} random signals...")
    signals = [generate_random_signal(sample_rate, duration) for _ in range(n_signals)]
    print(f"Generated {len(signals)} signals")

    # Set up parameters
    stft = sg.StftParams(
        n_fft=512, hop_size=256, window=sg.WindowType.hanning, centre=True
    )
    params = sg.SpectrogramParams(stft, sample_rate=sample_rate)
    mel_params = sg.MelParams(n_mels=80, f_min=0.0, f_max=8000.0)
    db_params = sg.LogParams(floor_db=-80.0)

    # ========================================================================
    # Method 1: Without plan reuse (convenience API)
    # ========================================================================
    print("\n" + "=" * 60)
    print("Method 1: Without Plan Reuse (Convenience API)")
    print("=" * 60)

    print("\nProcessing signals...")
    start_time = time.time()

    results_no_reuse = []
    for i, signal in enumerate(signals):
        spec = sg.compute_mel_db_spectrogram(signal, params, mel_params, db_params)
        results_no_reuse.append(spec)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n_signals} signals...")

    time_no_reuse = time.time() - start_time

    print(f"\nCompleted in {time_no_reuse:.3f} seconds")
    print(f"  Average: {time_no_reuse / n_signals * 1000:.2f} ms per signal")

    # ========================================================================
    # Method 2: With plan reuse (planner API)
    # ========================================================================
    print("\n" + "=" * 60)
    print("Method 2: With Plan Reuse (Planner API)")
    print("=" * 60)

    print("\nCreating plan...")
    planner = sg.SpectrogramPlanner()
    plan = planner.mel_db_plan(params, mel_params, db_params)
    print("Plan created")

    print("\nProcessing signals...")
    start_time = time.time()

    results_with_reuse = []
    for i, signal in enumerate(signals):
        spec = plan.compute(signal)
        results_with_reuse.append(spec)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n_signals} signals...")

    time_with_reuse = time.time() - start_time

    print(f"\nCompleted in {time_with_reuse:.3f} seconds")
    print(f"  Average: {time_with_reuse / n_signals * 1000:.2f} ms per signal")

    # ========================================================================
    # Performance comparison
    # ========================================================================
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)

    speedup = time_no_reuse / time_with_reuse
    time_saved = time_no_reuse - time_with_reuse
    percent_faster = (1 - time_with_reuse / time_no_reuse) * 100

    print(f"\nWithout plan reuse: {time_no_reuse:.3f} s")
    print(f"With plan reuse:    {time_with_reuse:.3f} s")
    print(f"\nSpeedup:            {speedup:.2f}x faster")
    print(f"Time saved:         {time_saved:.3f} s ({percent_faster:.1f}% faster)")

    # Verify results are identical
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    print("\nVerifying that both methods produce identical results...")

    all_match = True
    for i, (spec1, spec2) in enumerate(zip(results_no_reuse, results_with_reuse)):
        if spec1.shape != spec2.shape:
            print(f"  Signal {i}: Shape mismatch!")
            all_match = False
            continue

        if not np.allclose(spec1.data, spec2.data, rtol=1e-10):
            print(f"  Signal {i}: Data mismatch!")
            all_match = False
            continue

    if all_match:
        print("All results match perfectly!")
    else:
        print("✗ Some results don't match")

    # ========================================================================
    # Recommendation
    # ========================================================================
    print("\n" + "=" * 60)
    print("Recommendation")
    print("=" * 60)

    print("\nFor batch processing:")
    print("  • Use the planner API (SpectrogramPlanner + plans)")
    print(f"  • In this example: {speedup:.1f}x faster")
    print("  • Benefit increases with FFT size and batch size")
    print("  • Especially important for real-time processing")

    print("\nFor single computations:")
    print("  • Use convenience functions (compute_*)")
    print("  • Simpler API, no plan management")
    print("  • Minimal overhead for one-off computations")

    print("\nBatch processing example completed!")


if __name__ == "__main__":
    main()
