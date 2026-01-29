#!/usr/bin/env python3
"""
Streaming/Frame-by-Frame Processing Example

Demonstrates real-time processing using frame-by-frame computation.
Useful for online applications where you need to process audio as it arrives.
"""

import numpy as np
import spectrograms as sg


def simulate_streaming_audio(sample_rate, duration, chunk_duration=0.1):
    """
    Simulate streaming audio by yielding chunks.

    Args:
        sample_rate: Sample rate in Hz
        duration: Total duration in seconds
        chunk_duration: Duration of each chunk in seconds

    Yields:
        Audio chunks as numpy arrays
    """
    chunk_samples = int(sample_rate * chunk_duration)
    total_samples = int(sample_rate * duration)

    # Generate a frequency-modulated signal
    t_all = np.linspace(0, duration, total_samples, dtype=np.float64)
    # Frequency varies from 200 Hz to 800 Hz over time
    f_t = 200 + 600 * (t_all / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / sample_rate
    signal = np.sin(phase)

    # Yield chunks
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        yield signal[start:end]


def main():
    print("=" * 60)
    print("Streaming/Frame-by-Frame Processing Example")
    print("=" * 60)

    # Configuration
    sample_rate = 16000
    total_duration = 2.0
    chunk_duration = 0.1  # Process in 100ms chunks

    print(f"\nStreaming configuration:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Total duration: {total_duration} s")
    print(f"  Chunk duration: {chunk_duration} s")
    print(f"  Chunk size: {int(sample_rate * chunk_duration)} samples")

    # Set up spectrogram parameters
    stft = sg.StftParams(n_fft=512, hop_size=256, window="hanning", centre=True)
    params = sg.SpectrogramParams(stft, sample_rate=sample_rate)
    mel_params = sg.MelParams(n_mels=40, f_min=0.0, f_max=8000.0)

    # Create plan for efficient processing
    planner = sg.SpectrogramPlanner()
    plan = planner.mel_power_plan(params, mel_params)

    print(f"\nSTFT parameters:")
    print(f"  FFT size: {stft.n_fft}")
    print(f"  Hop size: {stft.hop_size}")
    print(f"  Mel bands: {mel_params.n_mels}")

    # ========================================================================
    # Method 1: Batch processing (compute entire spectrogram at once)
    # ========================================================================
    print("\n" + "=" * 60)
    print("Method 1: Batch Processing (for comparison)")
    print("=" * 60)

    # Collect all audio first
    all_chunks = list(
        simulate_streaming_audio(sample_rate, total_duration, chunk_duration)
    )
    full_signal = np.concatenate(all_chunks)

    print(f"\nProcessing full signal at once...")
    print(f"  Total samples: {len(full_signal)}")

    batch_spec = plan.compute(full_signal)

    print(f"\nBatch spectrogram computed:")
    print(f"  Shape: {batch_spec.shape}")
    print(f"  Duration: {batch_spec.duration():.3f} s")

    # ========================================================================
    # Method 2: Frame-by-frame streaming processing
    # ========================================================================
    print("\n" + "=" * 60)
    print("Method 2: Frame-by-Frame Streaming")
    print("=" * 60)

    # For streaming, we need to buffer audio to compute frames
    # Each frame needs n_fft samples, and we hop by hop_size samples

    frame_buffer = np.array([], dtype=np.float64)
    frame_results = []
    chunks_processed = 0

    print(f"\nStreaming audio chunks...")

    for chunk in simulate_streaming_audio(sample_rate, total_duration, chunk_duration):
        chunks_processed += 1

        # Add chunk to buffer
        frame_buffer = np.concatenate([frame_buffer, chunk])

        # Compute as many complete frames as possible
        frames_in_buffer = (len(frame_buffer) - stft.n_fft) // stft.hop_size + 1

        if frames_in_buffer > 0:
            print(
                f"  Chunk {chunks_processed}: {len(chunk)} samples → {frames_in_buffer} frames"
            )

            for frame_idx in range(frames_in_buffer):
                # Compute single frame
                frame_data = plan.compute_frame(frame_buffer, frame_idx)
                frame_results.append(frame_data)

            # Remove processed samples from buffer (keep overlap for next frames)
            samples_to_keep = stft.n_fft + (frames_in_buffer - 1) * stft.hop_size
            frame_buffer = frame_buffer[len(frame_buffer) - samples_to_keep :]

    print(f"\nStreaming complete:")
    print(f"  Chunks processed: {chunks_processed}")
    print(f"  Frames computed: {len(frame_results)}")
    print(f"  Buffer remaining: {len(frame_buffer)} samples")

    # ========================================================================
    # Verification
    # ========================================================================
    print("\n" + "=" * 60)
    print("Verification: Streaming vs Batch")
    print("=" * 60)

    # Convert streaming frames to array
    streaming_spec = np.column_stack(frame_results)

    print(f"\nBatch spectrogram shape:     {batch_spec.data.shape}")
    print(f"Streaming spectrogram shape: {streaming_spec.shape}")

    # Compare the overlapping portion
    n_compare_frames = min(batch_spec.n_frames, streaming_spec.shape[1])

    batch_frames = batch_spec.data[:, :n_compare_frames]
    streaming_frames = streaming_spec[:, :n_compare_frames]

    # Check if results match
    matches = np.allclose(batch_frames, streaming_frames, rtol=1e-10)

    print(f"\nComparing first {n_compare_frames} frames:")
    print(f"  Results match: {matches}")

    if matches:
        print("  Streaming and batch processing produce identical results!")
    else:
        max_diff = np.max(np.abs(batch_frames - streaming_frames))
        print(f"  ✗ Maximum difference: {max_diff:.2e}")

    # ========================================================================
    # Use case: Real-time feature extraction
    # ========================================================================
    print("\n" + "=" * 60)
    print("Use Case: Real-Time Feature Extraction")
    print("=" * 60)

    print("\nSimulating real-time processing...")

    # Process the signal again, but this time extract features from each chunk
    frame_buffer = np.array([], dtype=np.float64)
    chunk_num = 0

    for chunk in simulate_streaming_audio(sample_rate, total_duration, chunk_duration):
        chunk_num += 1
        frame_buffer = np.concatenate([frame_buffer, chunk])

        frames_in_buffer = (len(frame_buffer) - stft.n_fft) // stft.hop_size + 1

        if frames_in_buffer > 0:
            # Extract features from the latest frame
            latest_frame = plan.compute_frame(frame_buffer, frames_in_buffer - 1)

            # Example feature: spectral centroid (weighted mean frequency)
            frequencies = np.array(batch_spec.frequencies)[: len(latest_frame)]
            spectral_centroid = np.sum(frequencies * latest_frame) / (
                np.sum(latest_frame) + 1e-10
            )

            # Example feature: spectral rolloff (95% of energy)
            cumsum = np.cumsum(latest_frame)
            rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0][0]
            spectral_rolloff = frequencies[rolloff_idx]

            print(
                f"  Chunk {chunk_num}: centroid={spectral_centroid:.1f} Hz, rolloff={spectral_rolloff:.1f} Hz"
            )

            # Keep necessary samples for next frames
            samples_to_keep = stft.n_fft + (frames_in_buffer - 1) * stft.hop_size
            frame_buffer = frame_buffer[len(frame_buffer) - samples_to_keep :]

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print("\nWhen to use frame-by-frame processing:")
    print("  • Real-time audio analysis")
    print("  • Online feature extraction")
    print("  • Low-latency applications")
    print("  • Streaming audio sources")
    print("  • Memory-constrained environments")

    print("\nWhen to use batch processing:")
    print("  • Offline analysis of complete audio files")
    print("  • When you need to look ahead/behind in time")
    print("  • Simpler code, no buffer management")
    print("  • Slightly more efficient for large files")

    print("\nPerformance tip:")
    print("  • Always use a plan (SpectrogramPlanner) for repeated computations")
    print("  • Reuse the plan for all frames in your streaming application")

    print("\nStreaming example completed!")


if __name__ == "__main__":
    main()
