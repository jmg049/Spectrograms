#!/usr/bin/env python3
"""
Basic 2D FFT Example

Demonstrates:
- Computing 2D FFT of an image
- Power and magnitude spectra
- Roundtrip verification (FFT -> IFFT)
- Using fftshift for visualization
- Batch processing with Fft2dPlanner
"""

import numpy as np
import spectrograms as sg


def main():
    print("=== Basic 2D FFT Example ===\n")

    # Create a test image with known frequency content
    # Combination of sine waves in different directions
    nrows, ncols = 128, 128
    print(f"Image size: {nrows} x {ncols}")

    y, x = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing="ij")
    image = (
        np.sin(2 * np.pi * x / 32)  # Horizontal stripes
        + np.sin(2 * np.pi * y / 16)  # Vertical stripes
        + 0.5 * np.sin(2 * np.pi * (x + y) / 40)  # Diagonal pattern
    )

    print(f"Image range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"Image mean: {image.mean():.2f}\n")

    # === 1. Basic 2D FFT ===
    print("1. Computing 2D FFT...")
    spectrum = sg.fft2d(image)
    print(f"   Spectrum shape: {spectrum.shape}")
    print(f"   (Note: {ncols}//2 + 1 = {ncols // 2 + 1} due to Hermitian symmetry)")
    print(f"   DC component magnitude: {abs(spectrum[0, 0]):.2f}\n")

    # === 2. Power Spectrum ===
    print("2. Computing power spectrum...")
    power = sg.power_spectrum_2d(image)
    print(f"   Power shape: {power.shape}")
    print(f"   Total power: {power.sum():.2e}")
    print(f"   DC power: {power[0, 0]:.2e}\n")

    # === 3. Magnitude Spectrum ===
    print("3. Computing magnitude spectrum...")
    magnitude = sg.magnitude_spectrum_2d(image)
    print(f"   Magnitude shape: {magnitude.shape}")
    print(f"   Max magnitude: {magnitude.max():.2e}\n")

    # === 4. Roundtrip Test ===
    print("4. Testing roundtrip (FFT -> IFFT)...")
    reconstructed = sg.ifft2d(spectrum, ncols)

    max_error = np.abs(reconstructed - image).max()
    mean_error = np.abs(reconstructed - image).mean()

    print(f"   Reconstructed shape: {reconstructed.shape}")
    print(f"   Max error: {max_error:.2e}")
    print(f"   Mean error: {mean_error:.2e}")

    if max_error < 1e-10:
        print("   Roundtrip successful (error < 1e-10)\n")
    else:
        print("   ✗ Roundtrip failed (error too large)\n")

    # === 5. FFTShift for Visualization ===
    print("5. Using fftshift to center DC component...")

    # Compute power spectrum and shift
    power_shifted = sg.fftshift(power)

    print("   Original DC location: (0, 0)")
    print(f"   Shifted DC location: (~{nrows // 2}, ~{power.shape[1] // 2})")
    print(
        f"   Center value in shifted spectrum: {power_shifted[nrows // 2, power.shape[1] // 2]:.2e}\n"
    )

    # === 6. Batch Processing with Planner ===
    print("6. Batch processing with Fft2dPlanner...")

    # Create multiple test images
    n_images = 5
    images = [np.random.randn(64, 64) for _ in range(n_images)]

    # Use planner for efficient batch processing
    planner = sg.Fft2dPlanner()

    print(f"   Processing {n_images} images...")
    spectra = []
    for i, img in enumerate(images):
        spec = planner.fft2d(img)
        spectra.append(spec)
        print(f"   Image {i + 1}: shape {img.shape} -> spectrum {spec.shape}")

    print("\n   Batch processing complete")

    # Verify power spectra
    print("\n7. Computing power spectra with planner...")
    for i, img in enumerate(images):
        power = planner.power_spectrum_2d(img)
        print(f"   Image {i + 1} total power: {power.sum():.2e}")

    print("\n=== Example Complete ===")
    print("\nKey takeaways:")
    print("- 2D FFT produces (nrows, ncols/2+1) output due to Hermitian symmetry")
    print("- Roundtrip FFT->IFFT preserves original data (< 1e-10 error)")
    print("- fftshift centers DC component for visualization")
    print("- Fft2dPlanner reuses plans for efficient batch processing")


if __name__ == "__main__":
    main()
