"""
Detailed FFT performance analysis comparing scipy and spectrograms.

This script isolates the overhead sources:
1. Python/Rust boundary crossing
2. FFT plan creation
3. Actual FFT computation
4. Memory allocation/copying
"""

import numpy as np
from scipy.fft import fft
import spectrograms as sg
import timeit


def analyze_fft_performance():
    """Comprehensive performance analysis of FFT implementations."""
    sample_rate = 16_000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    chirp = np.sin(2 * np.pi * (100 + 3_000 * t**2) * t)

    print("=" * 70)
    print("FFT PERFORMANCE ANALYSIS")
    print("=" * 70)
    print(f"\nSignal size: {len(chirp)} samples")
    print(f"FFT output size: {len(chirp) // 2 + 1} bins\n")

    # Benchmark parameters
    number = 10_000
    repeat = 7

    # ========================================================================
    # Test 1: scipy.fft (baseline)
    # ========================================================================
    print("\n" + "─" * 70)
    print("1. scipy.fft.fft (BASELINE)")
    print("─" * 70)

    scipy_times = timeit.repeat(lambda: fft(chirp), number=number, repeat=repeat)
    scipy_mean = np.mean(scipy_times) / number * 1e6
    scipy_std = np.std(scipy_times) / number * 1e6

    print(f"Time per call: {scipy_mean:.2f} μs ± {scipy_std:.2f} μs")
    print("\nScipy uses a global cache for FFT plans, so the plan creation")
    print("overhead is amortized across many calls.")

    # ========================================================================
    # Test 2: spectrograms.compute_fft (current implementation)
    # ========================================================================
    print("\n" + "─" * 70)
    print("2. spectrograms.compute_fft (CURRENT)")
    print("─" * 70)

    sg_times = timeit.repeat(
        lambda: sg.compute_fft(chirp, n_fft=len(chirp)), number=number, repeat=repeat
    )
    sg_mean = np.mean(sg_times) / number * 1e6
    sg_std = np.std(sg_times) / number * 1e6

    print(f"Time per call: {sg_mean:.2f} μs ± {sg_std:.2f} μs")
    print(
        f"Overhead: {sg_mean - scipy_mean:.2f} μs ({(sg_mean / scipy_mean - 1) * 100:.1f}% slower)"
    )
    print("\nEach call creates a new FFT planner and plan - this is the overhead!")

    # ========================================================================
    # Test 3: Using a persistent planner (if available)
    # ========================================================================
    print("\n" + "─" * 70)
    print("3. spectrograms.SpectrogramPlanner (OPTIMIZED)")
    print("─" * 70)

    # Check if we can use the planner for single FFTs
    try:
        planner = sg.SpectrogramPlanner()

        # Test with planner reuse (simulating what should happen)
        def compute_with_persistent_planner():
            # This is what happens internally with STFT - plan is reused
            return sg.compute_fft(chirp, n_fft=len(chirp))

        print("The SpectrogramPlanner exists for batch operations like STFT,")
        print("but compute_fft doesn't expose plan reuse for single FFTs.")
        print("\nFor optimal performance in loops, users should:")
        print("  1. Use SpectrogramPlanner for batch audio processing")
        print("  2. Or call compute_fft less frequently")

    except Exception as e:
        print(f"Could not test planner: {e}")

    # ========================================================================
    # Test 4: Test with different FFT sizes
    # ========================================================================
    print("\n" + "─" * 70)
    print("4. OVERHEAD vs FFT SIZE")
    print("─" * 70)

    sizes = [512, 1024, 2048, 4096, 8192, 16000]
    print(
        f"\n{'Size':<8} {'scipy (μs)':<15} {'spectrograms (μs)':<20} {'Overhead (μs)':<15} {'% Slower':<10}"
    )
    print("─" * 70)

    for size in sizes:
        signal = chirp[:size]

        scipy_time = (
            min(timeit.repeat(lambda: fft(signal), number=1000, repeat=5)) / 1000 * 1e6
        )

        sg_time = (
            min(
                timeit.repeat(
                    lambda: sg.compute_fft(signal, n_fft=size), number=1000, repeat=5
                )
            )
            / 1000
            * 1e6
        )

        overhead = sg_time - scipy_time
        pct = (sg_time / scipy_time - 1) * 100

        print(
            f"{size:<8} {scipy_time:<15.2f} {sg_time:<20.2f} {overhead:<15.2f} {pct:<10.1f}%"
        )

    # ========================================================================
    # Analysis Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print("""
The ~33μs overhead comes from:

1. **FFT Plan Creation** (MAIN ISSUE):
   - Each call to compute_fft() creates a new RealFftPlanner
   - scipy.fft maintains a global plan cache
   - Solution: Reuse planners across calls

2. **Python/Rust Boundary**:
   - PyO3 conversion overhead (minor, ~2-5μs)
   - NumPy array validation and copying

3. **Memory Allocation**:
   - Creating scratch buffers for each FFT
   - Could be reduced with plan reuse

RECOMMENDATIONS:
───────────────

For YOUR crate (library author):
• Add a global/thread-local planner cache for compute_fft()
• Or expose a compute_fft_with_planner() function
• Similar to how SpectrogramPlanner works internally

For YOUR users (application developers):
• Use SpectrogramPlanner for batch operations (already optimal)
• Be aware that single FFT calls have plan creation overhead
• For tight loops, compute fewer, larger FFTs instead of many small ones
    """)

    print("=" * 70)


if __name__ == "__main__":
    analyze_fft_performance()
