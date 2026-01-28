#!/usr/bin/env python3
"""
Image Blur via FFT Convolution

Demonstrates:
- Creating Gaussian kernels
- FFT-based convolution (faster for large kernels)
- Comparing blur levels
- Analyzing frequency content changes
"""

import numpy as np
import spectrograms as sg

def main():
    print("=== Image Blur via FFT Convolution ===\n")

    # Create a test image with sharp features
    size = 256
    print(f"Image size: {size} x {size}")

    # Create image with sharp edges and patterns
    image = np.zeros((size, size))

    # Add some rectangles
    image[50:100, 50:100] = 1.0
    image[150:200, 100:200] = 0.8

    # Add a checkerboard pattern
    for i in range(0, size, 32):
        for j in range(0, size, 32):
            if (i + j) % 64 == 0:
                image[i:i+16, j:j+16] = 0.5

    print(f"Original image range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"Original image mean: {image.mean():.3f}\n")

    # === 1. Create Gaussian Kernels ===
    print("1. Creating Gaussian blur kernels...")

    blur_levels = [
        (5, 1.0, "Light"),
        (9, 2.0, "Medium"),
        (15, 3.0, "Heavy"),
    ]

    kernels = []
    for size_k, sigma, name in blur_levels:
        kernel = sg.gaussian_kernel_2d(size_k, sigma)
        kernels.append((kernel, name))
        print(f"   {name} blur: kernel size {size_k}x{size_k}, sigma={sigma}")
        print(f"     Kernel sum: {kernel.sum():.6f} (should be ~1.0)")
        print(f"     Kernel max: {kernel.max():.6f}")

    print()

    # === 2. Apply Blurs via FFT Convolution ===
    print("2. Applying blurs via FFT convolution...")

    blurred_images = []
    for kernel, name in kernels:
        blurred = sg.convolve_fft(image, kernel)
        blurred_images.append((blurred, name))

        print(f"   {name} blur:")
        print(f"     Output range: [{blurred.min():.3f}, {blurred.max():.3f}]")
        print(f"     Output mean: {blurred.mean():.3f}")

    print()

    # === 3. Analyze Frequency Content ===
    print("3. Analyzing frequency content changes...")

    # Original image spectrum
    original_power = sg.power_spectrum_2d(image)
    original_total_power = original_power.sum()

    print(f"   Original image:")
    print(f"     Total power: {original_total_power:.2e}")
    print(f"     DC component: {original_power[0, 0]:.2e}")

    # Blurred images spectra
    for blurred, name in blurred_images:
        blurred_power = sg.power_spectrum_2d(blurred)
        blurred_total_power = blurred_power.sum()

        # Compare power in different frequency bands
        # Low frequencies (DC + nearby)
        low_freq_power = blurred_power[:5, :3].sum()
        high_freq_power = blurred_power[20:, 20:].sum() if blurred_power.shape[0] > 20 else 0

        print(f"\n   {name} blur:")
        print(f"     Total power: {blurred_total_power:.2e}")
        print(f"     Power ratio: {blurred_total_power / original_total_power:.3f}")
        print(f"     Low freq power: {low_freq_power:.2e}")
        print(f"     High freq power: {high_freq_power:.2e}")

    print()

    # === 4. Compare Statistics ===
    print("4. Comparing blur statistics...")

    print(f"\n   {'Blur Level':<15} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std Dev':<10}")
    print(f"   {'-'*60}")

    # Original
    print(f"   {'Original':<15} {image.min():<10.3f} {image.max():<10.3f} "
          f"{image.mean():<10.3f} {image.std():<10.3f}")

    # Blurred versions
    for blurred, name in blurred_images:
        print(f"   {name + ' blur':<15} {blurred.min():<10.3f} {blurred.max():<10.3f} "
              f"{blurred.mean():<10.3f} {blurred.std():<10.3f}")

    print()

    # === 5. Direct Comparison ===
    print("5. Effect of blur on sharp edges...")

    # Sample a line through a sharp edge
    row = 75  # Middle of first rectangle
    original_line = image[row, 40:110]
    print(f"\n   Original edge profile (row {row}, cols 40-110):")
    print(f"     Max gradient: {np.abs(np.diff(original_line)).max():.3f}")

    for blurred, name in blurred_images:
        blurred_line = blurred[row, 40:110]
        max_gradient = np.abs(np.diff(blurred_line)).max()
        print(f"   {name} blur:")
        print(f"     Max gradient: {max_gradient:.3f} (edge smoothness)")

    print("\n=== Example Complete ===")
    print("\nKey observations:")
    print("- FFT convolution preserves total image energy")
    print("- Heavier blurs reduce high-frequency content")
    print("- Sharp edges become smoother with increased blur")
    print("- Kernel sum = 1.0 ensures brightness preservation")
    print("\nFFT convolution is faster than spatial convolution for kernels > 7x7")


if __name__ == "__main__":
    main()
