import numpy as np
import spectrograms as sg

# 1 second of 440 Hz sine wave
sample_rate = 16000
t = np.linspace(0, 1, sample_rate, dtype=np.float64)
samples = np.sin(2 * np.pi * 440 * t)


# Configure parameters
stft = sg.StftParams(n_fft=512, hop_size=256, window=sg.WindowType.hanning, centre=True)

params = sg.SpectrogramParams(stft, sample_rate=sample_rate)

# Compute power spectrogram
spec = sg.compute_linear_power_spectrogram(samples, params)

print(f"Shape: {spec.n_bins} bins x {spec.n_frames} frames")

stft = sg.StftParams(512, 256, sg.WindowType.hanning, True)
params = sg.SpectrogramParams(stft, 16000)

# Mel filterbank
mel = sg.MelParams(n_mels=80, f_min=0.0, f_max=8000.0)

# dB scaling
db = sg.LogParams(floor_db=-80.0)

# Compute mel spectrogram in dB
spec = sg.compute_mel_db_spectrogram(samples, params, mel, db)

# Access data
print(f"Mel bands: {spec.n_bins}")
print(f"Frames: {spec.n_frames}")
print(f"Frequency range: {spec.frequency_range()}")

signals = [
    np.random.randn(16000),
    np.random.randn(16000),
    np.random.randn(16000),
]

stft = sg.StftParams(512, 256, sg.WindowType.hanning, True)
params = sg.SpectrogramParams(stft, 16000)
mel = sg.MelParams(80, 0.0, 8000.0)
db = sg.LogParams(-80.0)

# Create plan once
planner = sg.SpectrogramPlanner()
plan = planner.mel_db_plan(params, mel, db)

# Reuse for all signals (much faster!)
for signal in signals:
    spec = plan.compute(signal)
    # Process spec...


# Create a 256x256 image
image = np.zeros((256, 256), dtype=np.float64)
for i in range(256):
    for j in range(256):
        image[i, j] = np.sqrt((i - 128) ** 2 + (j - 128) ** 2)

# Compute 2D FFT
spectrum = sg.fft2d(image)
print(f"Spectrum shape: {spectrum.shape}")
# Output: (256, 129) due to Hermitian symmetry

# Apply Gaussian blur via FFT
kernel = sg.gaussian_kernel_2d(9, 2.0)
blurred = sg.convolve_fft(image, kernel)

# Apply high-pass filter for edge detection
edges = sg.highpass_filter(image, 0.1)

# Compute power spectrum
power = sg.power_spectrum_2d(image)

images = [
    np.random.randn(256, 256),
    np.random.randn(256, 256),
    np.random.randn(256, 256),
]

# Create planner once
planner = sg.Fft2dPlanner()

# Reuse for all images (faster!)
for image in images:
    spectrum = planner.fft2d(image)
    power = np.abs(spectrum) ** 2
    # Process power spectrum...


stft = sg.StftParams(512, 160, sg.WindowType.hanning, True)
mfcc_params = sg.MfccParams(n_mfcc=13)

mfccs = sg.compute_mfcc(
    samples, stft, sample_rate=16000, n_mels=40, mfcc_params=mfcc_params
)

# Shape: (13, n_frames)
print(f"MFCCs: {mfccs.shape}")


stft = sg.StftParams(4096, 512, sg.WindowType.hanning, True)
chroma_params = sg.ChromaParams.music_standard()

chroma = sg.compute_chromagram(
    samples, stft, sample_rate=22050, chroma_params=chroma_params
)

# Shape: (12, n_frames)
print(f"Chroma: {chroma.shape}")

# Use class methods
window = sg.WindowType.hanning
kaiser = sg.WindowType.kaiser(beta=8.0)
gauss = sg.WindowType.gaussian(std=0.4)

# Speech processing preset
params = sg.SpectrogramParams.speech_default(sample_rate=16000)

# Music processing preset
params = sg.SpectrogramParams.music_default(sample_rate=44100)

spec = sg.compute_linear_power_spectrogram(samples, params)

# Dimensions
n_bins = spec.n_bins
n_frames = spec.n_frames

# Data (numpy array)
data = spec.data  # shape: (n_bins, n_frames)

# Axes
freqs = spec.frequencies
times = spec.times
f_min, f_max = spec.frequency_range()
duration = spec.duration()

# Original parameters
params = spec.params
