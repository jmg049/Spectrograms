import numpy as np
import spectrograms as sg
import matplotlib.pyplot as plt

# Load or create image
image = np.random.randn(256, 256)

# Apply Gaussian blur to reduce noise
kernel = sg.gaussian_kernel_2d(5, 1.0)
denoised = sg.convolve_fft(image, kernel)

# Detect edges
edges = sg.detect_edges_fft(denoised)

# Sharpen the result
enhanced = sg.sharpen_fft(edges, amount=1.2)

# Display results
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original")
axes[1].imshow(denoised, cmap="gray")
axes[1].set_title("Denoised")
axes[2].imshow(edges, cmap="gray")
axes[2].set_title("Edges")
axes[3].imshow(enhanced, cmap="gray")
axes[3].set_title("Enhanced")
plt.show()
