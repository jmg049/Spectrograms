//! Spectro-Temporal Modulation Transfer Function (STMTF) Example
//!
//! This example demonstrates computing a 2D FFT on a spectrogram to obtain
//! the spectro-temporal modulation transfer function, which is useful in
//! auditory neuroscience research.
//!
//! The STMTF reveals how energy is distributed across:
//! - Spectral modulation (vertical): rate of change across frequency
//! - Temporal modulation (horizontal): rate of change over time

use non_empty_slice::NonEmptySlice;
use spectrograms::fft2d::{fftfreq, fftshift, magnitude_spectrum_2d};
use spectrograms::*;
use std::f64::consts::PI;

fn main() -> SpectrogramResult<()> {
    println!("=== Spectro-Temporal Modulation Transfer Function (STMTF) ===\n");

    // Generate test signal: amplitude-modulated tone
    let sample_rate = 16000.0;
    let duration = 2.0;
    let n_samples = (sample_rate * duration) as usize;

    // Carrier: 1000 Hz tone with 10 Hz amplitude modulation
    let carrier_freq = 1000.0;
    let mod_freq = 10.0;

    println!("Signal parameters:");
    println!("  Sample rate: {} Hz", sample_rate);
    println!("  Duration: {} s", duration);
    println!("  Carrier frequency: {} Hz", carrier_freq);
    println!("  Modulation frequency: {} Hz", mod_freq);
    println!();

    let samples: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            let am = 1.0 + 0.5 * (2.0 * PI * mod_freq * t).cos();
            am * (2.0 * PI * carrier_freq * t).sin()
        })
        .collect();
    let samples = NonEmptySlice::new(&samples).unwrap();
    // Compute mel spectrogram
    println!("Computing mel spectrogram...");
    let stft = StftParams::new(nzu!(512), nzu!(128), WindowType::Hanning, true)?;
    let params = SpectrogramParams::new(stft, sample_rate)?;
    let mel = MelParams::new(nzu!(64), 0.0, 8000.0)?;

    let spectrogram = MelPowerSpectrogram::compute(samples, &params, &mel, None)?;

    println!(
        "Spectrogram shape: {} freq bins × {} time frames",
        spectrogram.n_bins(),
        spectrogram.n_frames()
    );
    println!(
        "Frequency range: {:.1} - {:.1} Hz",
        spectrogram.frequency_range().0,
        spectrogram.frequency_range().1
    );
    println!("Duration: {:.2} s\n", spectrogram.duration());

    // Compute STMTF using 2D FFT
    // The spectrogram derefs to &Array2<f64>, so we can pass it directly!
    println!("Computing STMTF via 2D FFT...");

    // Get magnitude spectrum directly (most common for visualization)
    // The spectrogram derefs to &Array2<f64> thanks to generic Deref trait
    let stmtf_magnitude = magnitude_spectrum_2d(&(*spectrogram).view())?;
    println!("Magnitude STMTF shape: {:?}", stmtf_magnitude.dim());

    // Shift zero-frequency to center for visualization
    let stmtf_centered = fftshift(stmtf_magnitude);
    println!("Centered STMTF shape: {:?}\n", stmtf_centered.dim());

    // Calculate modulation frequencies using library functions
    let freq_bins = spectrogram.n_bins().get();
    let time_frames = spectrogram.n_frames().get();

    // Spectral modulation frequencies (cycles per bin)
    let spectral_mod_freqs = fftfreq(freq_bins, 1.0);

    // Temporal modulation frequencies (Hz)
    // Use actual time step from spectrogram
    let times = spectrogram.times();
    let time_step = if times.len().get() > 1 {
        times[1] - times[0]
    } else {
        params.frame_period_seconds()
    };
    let temporal_mod_freqs = fftfreq(time_frames, time_step);

    println!("Modulation frequency ranges:");
    println!(
        "  Spectral: {:.2} to {:.2} cycles/bin",
        spectral_mod_freqs[0],
        spectral_mod_freqs[spectral_mod_freqs.len() - 1]
    );
    println!(
        "  Temporal: {:.2} to {:.2} Hz",
        temporal_mod_freqs[0],
        temporal_mod_freqs[temporal_mod_freqs.len() - 1]
    );
    println!();

    println!("Expected STMTF features:");
    println!(
        "  - Strong peak near {} Hz temporal modulation (AM frequency)",
        mod_freq
    );
    println!("  - Low spectral modulation (pure tone has stable spectrum)");
    println!();

    println!("Key insight:");
    println!("  The Spectrogram derefs to Array2<f64>, so it works seamlessly");
    println!("  with all 2D FFT functions - no wrapper functions needed!");
    println!("  Use magnitude_spectrum_2d(), fftshift(), and fftfreq() from the library.");

    Ok(())
}
