#!/usr/bin/env python3
"""
Image Edge Detection via Frequency Filtering

Demonstrates:
- High-pass filtering to emphasize edges
- Low-pass filtering to remove noise
- Band-pass filtering for specific frequencies
- Edge detection and sharpening operations
- Comparing different cutoff frequencies
"""

import numpy as np
import spectrograms as sg

def create_test_image(size=256):
    """Create a test image with various features."""
    image = np.zeros((size, size))

    # Large rectangle
    image[50:150, 50:150] = 0.8

    # Small circles (approximate)
    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')

    circle1 = ((x - 180)**2 + (y - 80)**2) < 20**2
    circle2 = ((x - 180)**2 + (y - 180)**2) < 15**2

    image[circle1] = 1.0
    image[circle2] = 0.6

    # Add some noise
    noise = np.random.randn(size, size) * 0.05
    image = np.clip(image + noise, 0, 1)

    return image


def main():
    print("=== Image Edge Detection via Frequency Filtering ===\n")

    # === 1. Create Test Image ===
    print("1. Creating test image with edges and noise...")
    image = create_test_image(256)

    print(f"   Image size: {image.shape}")
    print(f"   Range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"   Mean: {image.mean():.3f}")
    print(f"   Std dev: {image.std():.3f}\n")

    # === 2. Analyze Original Frequency Content ===
    print("2. Analyzing frequency content...")

    original_power = sg.power_spectrum_2d(image)
    total_power = original_power.sum()

    # Split into frequency bands
    nrows, ncols = original_power.shape
    dc_power = original_power[0, 0]
    low_freq = original_power[:10, :6].sum()  # Low frequencies
    mid_freq = original_power[10:30, 6:16].sum() if nrows > 30 else 0
    high_freq = original_power[30:, 16:].sum() if nrows > 30 else 0

    print(f"   Total power: {total_power:.2e}")
    print(f"   DC component: {dc_power:.2e} ({100*dc_power/total_power:.1f}%)")
    print(f"   Low frequencies: {low_freq:.2e} ({100*low_freq/total_power:.1f}%)")
    print(f"   Mid frequencies: {mid_freq:.2e} ({100*mid_freq/total_power:.1f}%)")
    print(f"   High frequencies: {high_freq:.2e} ({100*high_freq/total_power:.1f}%)\n")

    # === 3. Low-Pass Filter (Smoothing) ===
    print("3. Applying low-pass filters (suppress high frequencies)...")

    cutoffs = [0.3, 0.2, 0.1]
    for cutoff in cutoffs:
        smoothed = sg.lowpass_filter(image, cutoff)

        print(f"   Cutoff {cutoff:.1f}:")
        print(f"     Output range: [{smoothed.min():.3f}, {smoothed.max():.3f}]")
        print(f"     Output std: {smoothed.std():.3f} (lower = smoother)")

    print()

    # === 4. High-Pass Filter (Edge Enhancement) ===
    print("4. Applying high-pass filters (emphasize edges)...")

    for cutoff in cutoffs:
        edges = sg.highpass_filter(image, cutoff)

        # High-pass output can have negative values (edges are +/- around zero)
        edge_strength = np.abs(edges).mean()
        max_edge = np.abs(edges).max()

        print(f"   Cutoff {cutoff:.1f}:")
        print(f"     Output range: [{edges.min():.3f}, {edges.max():.3f}]")
        print(f"     Mean |edge|: {edge_strength:.3f}")
        print(f"     Max |edge|: {max_edge:.3f}")

    print()

    # === 5. Edge Detection ===
    print("5. Detecting edges with detect_edges_fft()...")

    edges = sg.detect_edges_fft(image)

    print(f"   Edge map range: [{edges.min():.3f}, {edges.max():.3f}]")
    print(f"   Mean absolute edge: {np.abs(edges).mean():.3f}")
    print(f"   Max edge magnitude: {np.abs(edges).max():.3f}")

    # Count strong edges (threshold at 0.1)
    strong_edges = np.abs(edges) > 0.1
    edge_pixels = strong_edges.sum()
    print(f"   Strong edge pixels (|edge| > 0.1): {edge_pixels} "
          f"({100*edge_pixels/edges.size:.1f}% of image)\n")

    # === 6. Image Sharpening ===
    print("6. Sharpening image with different amounts...")

    sharpen_amounts = [0.5, 1.0, 2.0]
    for amount in sharpen_amounts:
        sharpened = sg.sharpen_fft(image, amount)

        print(f"   Sharpen amount {amount:.1f}:")
        print(f"     Output range: [{sharpened.min():.3f}, {sharpened.max():.3f}]")
        print(f"     Output std: {sharpened.std():.3f} (higher = more contrast)")

        # Measure edge enhancement
        sharp_edges = sg.highpass_filter(sharpened, 0.2)
        edge_strength = np.abs(sharp_edges).mean()
        print(f"     Edge strength: {edge_strength:.3f}")

    print()

    # === 7. Band-Pass Filter ===
    print("7. Applying band-pass filters (isolate frequency range)...")

    bands = [
        (0.1, 0.3, "Low-mid"),
        (0.2, 0.5, "Mid"),
        (0.4, 0.8, "Mid-high"),
    ]

    for low, high, name in bands:
        try:
            filtered = sg.bandpass_filter(image, low, high)

            print(f"   {name} band [{low:.1f}, {high:.1f}]:")
            print(f"     Output range: [{filtered.min():.3f}, {filtered.max():.3f}]")
            print(f"     Output std: {filtered.std():.3f}")
        except Exception as e:
            print(f"   {name} band [{low:.1f}, {high:.1f}]: Error - {e}")

    print()

    # === 8. Noise Reduction ===
    print("8. Comparing original vs smoothed for noise reduction...")

    # Apply mild low-pass to reduce noise
    denoised = sg.lowpass_filter(image, 0.4)

    print(f"   Original:")
    print(f"     Std dev: {image.std():.3f} (includes noise)")

    print(f"   Denoised (lowpass 0.4):")
    print(f"     Std dev: {denoised.std():.3f} (reduced variance)")

    # Estimate noise as difference
    noise_estimate = image - denoised
    print(f"   Estimated noise:")
    print(f"     Std dev: {noise_estimate.std():.3f}")
    print(f"     (close to injected noise std of 0.05)")

    print("\n=== Example Complete ===")
    print("\nKey observations:")
    print("- Low-pass filters smooth images by removing high frequencies (edges)")
    print("- High-pass filters emphasize edges by removing low frequencies")
    print("- Edge detection finds regions of rapid intensity change")
    print("- Sharpening enhances edges while preserving overall structure")
    print("- Band-pass filters isolate specific frequency ranges")
    print("- All operations work in frequency domain (fast for large images)")


if __name__ == "__main__":
    main()
