use non_empty_slice::NonEmptyVec;
/// Mel-frequency spectrogram example
///
/// This example demonstrates:
/// - Computing mel-frequency spectrograms
/// - Using logarithmic (dB) amplitude scaling
/// - Accessing mel-frequency axes
use spectrograms::{
    LogParams, MelDbSpectrogram, MelParams, SpectrogramParams, StftParams, WindowType, nzu,
};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate a more complex signal: two sine waves
    let sample_rate = 16000.0;
    let duration = 1.0;

    let samples: Vec<f64> = (0..(duration * sample_rate) as usize)
        .map(|i| {
            let t = i as f64 / sample_rate;
            // 440 Hz (A4) + 880 Hz (A5)
            (2.0 * PI * 440.0 * t).sin() + 0.5 * (2.0 * PI * 880.0 * t).sin()
        })
        .collect();
    let samples = NonEmptyVec::new(samples).unwrap();
    println!("Generated {} samples with two frequencies", samples.len());

    // Set up STFT parameters
    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    let params = SpectrogramParams::new(stft, sample_rate)?;

    // Set up mel filterbank parameters
    let mel = MelParams::new(
        nzu!(80), // n_mels: number of mel bands
        0.0,      // f_min: minimum frequency (Hz)
        8000.0,   // f_max: maximum frequency (Hz)
    )?;

    // Set up logarithmic (dB) scaling
    let db = LogParams::new(-80.0)?; // floor at -80 dB

    // Compute mel spectrogram in dB scale
    let spec = MelDbSpectrogram::compute(&samples, &params, &mel, Some(&db))?;

    println!("\nMel Spectrogram properties:");
    println!("  Mel bands: {}", spec.n_bins());
    println!("  Time frames: {}", spec.n_frames());

    // Access mel-frequency axis
    let mel_freqs = spec.axes().frequencies();
    println!("\nMel frequency range:");
    println!("  Lowest mel center: {:.1} Hz", mel_freqs[0]);
    println!(
        "  Highest mel center: {:.1} Hz",
        mel_freqs[mel_freqs.len().get() - 1]
    );

    // Print a few mel band center frequencies
    println!("\nFirst 10 mel band centers (Hz):");
    for (i, &freq) in mel_freqs.iter().take(10).enumerate() {
        println!("  Mel band {}: {:.1} Hz", i, freq);
    }

    // Find average energy per mel band (across all frames)
    println!("\nAverage energy per mel band (dB):");
    for i in (0..spec.n_bins().get()).step_by(spec.n_bins().get() / 5) {
        let row = spec.data().row(i);
        let avg = row.iter().sum::<f64>() / row.len() as f64;
        println!("  Mel band {} ({:.1} Hz): {:.2} dB", i, mel_freqs[i], avg);
    }

    Ok(())
}
