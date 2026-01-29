#!/usr/bin/env python3
"""
Focused benchmark: spectrograms vs librosa for Mel spectrogram computation.

Compares only Mel spectrogram since that's the primary operation librosa provides.
"""

import numpy as np
import time
import spectrograms as sg
import librosa

# Benchmark configuration
WARMUP = 10
RUNS = 100
SAMPLE_RATE = 16000

# Test configurations (representative subset)
CONFIGS = [
    # (n_fft, hop_length, n_mels, description)
    (512, 128, 80, "Small FFT, 80 mels"),
    (1024, 256, 128, "Medium FFT, 128 mels"),
    (2048, 512, 128, "Large FFT, 128 mels"),
]


# Test signals
def make_fixtures(sample_rate: int):
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    return {
        "sine_440": np.sin(2 * np.pi * 440 * t),
        "noise": np.random.randn(sample_rate),
        "chirp": np.sin(2 * np.pi * (100 + 3_000 * t**2) * t),
    }


FIXTURES = make_fixtures(SAMPLE_RATE)


def benchmark_fn(fn, warmup=WARMUP, runs=RUNS):
    """Run benchmark with warmup and return timing statistics."""
    # Warmup
    for _ in range(warmup):
        fn()

    # Timed runs
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # Convert to ms

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def spectrograms_mel(signal, sr, n_fft, hop_length, n_mels, norm=None):
    """Compute mel spectrogram using spectrograms."""
    stft_params = sg.StftParams(
        n_fft=n_fft, hop_size=hop_length, window=sg.WindowType.hanning, centre=True
    )
    spec_params = sg.SpectrogramParams(stft_params, sample_rate=sr)
    mel_params = sg.MelParams(n_mels=n_mels, f_min=0.0, f_max=sr / 2.0, norm=norm)

    return sg.compute_mel_power_spectrogram(signal, spec_params, mel_params)


def librosa_mel(signal, sr, n_fft, hop_length, n_mels, norm="slaney"):
    """Compute mel spectrogram using librosa."""
    return librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0.0,
        fmax=sr / 2.0,
        power=2.0,  # Power spectrogram
        center=True,
        window="hann",
        norm=norm,
    )


def run_comparison():
    """Run the benchmark comparison."""
    print("=" * 80)
    print("Mel Spectrogram Benchmark: spectrograms vs librosa")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print("  Signal duration: 1 second")
    print(f"  Warmup runs: {WARMUP}")
    print(f"  Timed runs: {RUNS}")
    print()

    results = []

    # Test both unnormalized and Slaney-normalized versions
    norm_tests = [
        (None, None, "Unnormalized"),
        ("slaney", "slaney", "Slaney-normalized (librosa default)"),
    ]

    for norm_sg, norm_librosa, norm_desc in norm_tests:
        print(f"\n{'=' * 80}")
        print(f"Mode: {norm_desc}")
        print(f"{'=' * 80}")

        for n_fft, hop_length, n_mels, description in CONFIGS:
            print(f"\n{description}")
            print(f"  n_fft={n_fft}, hop_length={hop_length}, n_mels={n_mels}")
            print("-" * 80)

            for fixture_name, signal in FIXTURES.items():
                # Benchmark spectrograms
                sg_time = benchmark_fn(
                    lambda: spectrograms_mel(
                        signal, SAMPLE_RATE, n_fft, hop_length, n_mels, norm=norm_sg
                    )
                )

                # Benchmark librosa
                librosa_time = benchmark_fn(
                    lambda: librosa_mel(
                        signal,
                        SAMPLE_RATE,
                        n_fft,
                        hop_length,
                        n_mels,
                        norm=norm_librosa,
                    )
                )

                speedup = librosa_time["mean"] / sg_time["mean"]

                print(
                    f"  {fixture_name:10s}: "
                    f"spectrograms={sg_time['mean']:6.3f}ms (±{sg_time['std']:5.3f})  "
                    f"librosa={librosa_time['mean']:6.3f}ms (±{librosa_time['std']:5.3f})  "
                    f"speedup={speedup:.2f}x"
                )

                results.append(
                    {
                        "norm": norm_desc,
                        "config": description,
                        "n_fft": n_fft,
                        "hop_length": hop_length,
                        "n_mels": n_mels,
                        "fixture": fixture_name,
                        "spectrograms_ms": sg_time["mean"],
                        "librosa_ms": librosa_time["mean"],
                        "speedup": speedup,
                    }
                )

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for norm_desc in ["Unnormalized", "Slaney-normalized (librosa default)"]:
        norm_results = [r for r in results if r["norm"] == norm_desc]
        if not norm_results:
            continue

        speedups = [r["speedup"] for r in norm_results]
        print(f"\n{norm_desc}:")
        print(f"  Mean speedup:   {np.mean(speedups):.2f}x")
        print(f"  Median speedup: {np.median(speedups):.2f}x")
        print(f"  Min speedup:    {np.min(speedups):.2f}x")
        print(f"  Max speedup:    {np.max(speedups):.2f}x")
        print(f"  Std:            {np.std(speedups):.2f}")

        # By configuration
        print("\n  Speedup by configuration:")
        for n_fft, hop_length, n_mels, description in CONFIGS:
            config_speedups = [
                r["speedup"]
                for r in norm_results
                if r["n_fft"] == n_fft and r["n_mels"] == n_mels
            ]
            if config_speedups:
                print(
                    f"    {description:25s}: {np.mean(config_speedups):.2f}x (±{np.std(config_speedups):.2f})"
                )

    # Verification (check that outputs are numerically close)
    print("\n" + "=" * 80)
    print("NUMERICAL VERIFICATION")
    print("=" * 80)

    n_fft, hop_length, n_mels = 1024, 256, 128
    signal = FIXTURES["sine_440"]

    # Test both unnormalized and Slaney-normalized
    for norm_sg, norm_librosa, norm_desc in [
        (None, None, "Unnormalized"),
        ("slaney", "slaney", "Slaney-normalized"),
    ]:
        print(f"\n{norm_desc}:")
        print("-" * 80)

        sg_result = spectrograms_mel(
            signal, SAMPLE_RATE, n_fft, hop_length, n_mels, norm=norm_sg
        )
        librosa_result = librosa_mel(
            signal, SAMPLE_RATE, n_fft, hop_length, n_mels, norm=norm_librosa
        )

        # spectrograms result is already a numpy array
        sg_array = np.array(sg_result)

        # Check shapes
        print(
            f"  Shapes: spectrograms={sg_array.shape}, librosa={librosa_result.shape}"
        )

        # Check numerical agreement
        if sg_array.shape == librosa_result.shape:
            abs_diff = np.abs(sg_array - librosa_result)
            rel_diff = abs_diff / (np.abs(librosa_result) + 1e-12)

            print(f"  Max absolute difference: {np.max(abs_diff):.2e}")
            print(f"  Mean absolute difference: {np.mean(abs_diff):.2e}")
            print(f"  Max relative difference: {np.max(rel_diff):.2e}")
            print(f"  Mean relative difference: {np.mean(rel_diff):.2e}")
        else:
            print("  ✗ Shape mismatch!")


if __name__ == "__main__":
    run_comparison()
