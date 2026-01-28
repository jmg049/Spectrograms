//! Example demonstrating serialization of spectrograms and related result types.
//!
//! This example shows how to:
//! - Compute spectrograms and other features
//! - Serialize them to JSON and binary formats (bincode)
//! - Deserialize them back
//! - Compare serialization sizes
//!
//! Run with: cargo run --example serde_example --features serde,realfft

#[cfg(feature = "serde")]
use spectrograms::*;

#[cfg(feature = "serde")]
use num_format::{Locale, ToFormattedString};

#[cfg(feature = "serde")]
fn run_example() -> Result<(), Box<dyn std::error::Error>> {
    use non_empty_slice::NonEmptyVec;

    println!("Spectrogram Serialization Example\n");
    println!("==================================\n");

    // Generate a test signal: 440 Hz sine wave
    let sample_rate = 16000.0;
    let duration = 2.0;
    let n_samples = (sample_rate * duration) as usize;
    let signal: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            (2.0 * std::f64::consts::PI * 440.0 * t).sin()
        })
        .collect();
    let signal = NonEmptyVec::new(signal).unwrap();
    println!("Generated {} samples at {} Hz", n_samples, sample_rate);
    println!();

    // 1. Linear Power Spectrogram
    println!("1. Linear Power Spectrogram");
    println!("----------------------------");
    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    let params = SpectrogramParams::new(stft, sample_rate)?;
    let spec = LinearPowerSpectrogram::compute(&signal, &params, None)?;

    println!(
        "Spectrogram shape: {} bins x {} frames",
        spec.n_bins(),
        spec.n_frames()
    );

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&spec)?;
    let json_size = json.len();
    println!("JSON size: {} bytes", json_size);

    // Serialize to bincode
    let bincode_bytes = bincode2::serialize(&spec)?;
    let bincode_size = bincode_bytes.len();
    println!("Bincode size: {} bytes", bincode_size);

    println!(
        "Bincode is {:.1}x smaller than JSON",
        json_size as f64 / bincode_size as f64
    );

    // Deserialize from JSON
    let deserialized: LinearPowerSpectrogram = serde_json::from_str(&json)?;
    println!(
        "Successfully deserialized from JSON: {} bins x {} frames",
        deserialized.n_bins(),
        deserialized.n_frames()
    );

    // Deserialize from bincode
    let deserialized_bin: LinearPowerSpectrogram = bincode2::deserialize(&bincode_bytes)?;
    println!(
        "Successfully deserialized from bincode: {} bins x {} frames",
        deserialized_bin.n_bins(),
        deserialized_bin.n_frames()
    );
    println!();

    // 2. Mel Spectrogram
    println!("2. Mel Spectrogram");
    println!("------------------");
    let mel_params = MelParams::new(nzu!(80), 0.0, 8000.0)?;
    let mel_spec = MelPowerSpectrogram::compute(&signal, &params, &mel_params, None)?;

    println!(
        "Mel spectrogram shape: {} bins x {} frames",
        mel_spec.n_bins(),
        mel_spec.n_frames()
    );

    let mel_json = serde_json::to_string_pretty(&mel_spec)?;
    let mel_json_size = mel_json.len();
    println!("JSON size: {} bytes", mel_json_size);

    let mel_bincode = bincode2::serialize(&mel_spec)?;
    let mel_bincode_size = mel_bincode.len();
    println!("Bincode size: {} bytes", mel_bincode_size);

    let mel_deserialized: MelPowerSpectrogram = serde_json::from_str(&mel_json)?;
    println!(
        "Successfully deserialized: {} mel bins, {} frames",
        mel_deserialized.n_bins(),
        mel_deserialized.n_frames()
    );
    println!();

    // 3. MFCC
    println!("3. MFCC Features");
    println!("----------------");
    let mfcc_stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    let mfcc_params = MfccParams::new(nzu!(13));
    let mfcc = mfcc(&signal, &mfcc_stft, sample_rate, nzu!(40), &mfcc_params)?;

    println!(
        "MFCC shape: {} coefficients x {} frames",
        mfcc.n_coefficients(),
        mfcc.n_frames()
    );

    let mfcc_json = serde_json::to_string_pretty(&mfcc)?;
    let mfcc_json_size = mfcc_json.len();
    println!("JSON size: {} bytes", mfcc_json_size);

    let mfcc_bincode = bincode2::serialize(&mfcc)?;
    let mfcc_bincode_size = mfcc_bincode.len();
    println!("Bincode size: {} bytes", mfcc_bincode_size);

    let mfcc_deserialized: Mfcc = bincode2::deserialize(&mfcc_bincode)?;
    println!(
        "Successfully deserialized: {} coefficients x {} frames",
        mfcc_deserialized.n_coefficients(),
        mfcc_deserialized.n_frames()
    );
    println!();

    // 4. Chromagram
    println!("4. Chromagram");
    println!("-------------");
    let chroma_stft = StftParams::new(nzu!(2048), nzu!(512), WindowType::Hanning, true)?;
    let chroma_params = ChromaParams::new(440.0, 55.0, 4200.0, ChromaNorm::L2)?;
    let chroma = chromagram(&signal, &chroma_stft, sample_rate, &chroma_params)?;

    println!(
        "Chromagram shape: {} pitch classes x {} frames",
        chroma.n_bins(),
        chroma.n_frames()
    );

    let chroma_json = serde_json::to_string_pretty(&chroma)?;
    let chroma_json_size = chroma_json.len();
    println!("JSON size: {} bytes", chroma_json_size);

    let chroma_bincode = bincode2::serialize(&chroma)?;
    let chroma_bincode_size = chroma_bincode.len();
    println!("Bincode size: {} bytes", chroma_bincode_size);

    let chroma_deserialized: Chromagram = serde_json::from_str(&chroma_json)?;
    println!(
        "Successfully deserialized: {} pitch classes x {} frames",
        chroma_deserialized.n_bins(),
        chroma_deserialized.n_frames()
    );
    println!();

    // 5. CQT Result
    println!("5. Constant-Q Transform");
    println!("-----------------------");
    let cqt_params = CqtParams::new(nzu!(12), nzu!(6), 55.0)?;
    let cqt = cqt(&signal, sample_rate, &cqt_params, nzu!(256))?;

    println!("CQT shape: {:?}", cqt.data.shape());
    println!("Frequency bins: {}", cqt.frequencies.len());

    let cqt_json = serde_json::to_string_pretty(&cqt)?;
    let cqt_json_size = cqt_json.len();
    println!("JSON size: {} bytes", cqt_json_size);

    let cqt_bincode = bincode2::serialize(&cqt)?;
    let cqt_bincode_size = cqt_bincode.len();
    println!("Bincode size: {} bytes", cqt_bincode_size);

    let cqt_deserialized: CqtResult = bincode2::deserialize(&cqt_bincode)?;
    println!(
        "Successfully deserialized: shape {:?}",
        cqt_deserialized.data.shape(),
    );
    println!();

    // Summary
    println!("=======");
    println!();
    println!("All result types successfully serialized and deserialized!");
    println!();
    println!("Size comparison (JSON vs Bincode):");
    println!(
        "  Linear Spectrogram: {} bytes vs {} bytes ({:.1}x)",
        json_size.to_formatted_string(&Locale::en),
        bincode_size.to_formatted_string(&Locale::en),
        json_size as f64 / bincode_size as f64
    );
    println!(
        "  Mel Spectrogram:    {} bytes vs {} bytes ({:.1}x)",
        mel_json_size.to_formatted_string(&Locale::en),
        mel_bincode_size.to_formatted_string(&Locale::en),
        mel_json_size as f64 / mel_bincode_size as f64
    );
    println!(
        "  MFCC:               {} bytes vs {} bytes ({:.1}x)",
        mfcc_json_size.to_formatted_string(&Locale::en),
        mfcc_bincode_size.to_formatted_string(&Locale::en),
        mfcc_json_size as f64 / mfcc_bincode_size as f64
    );
    println!(
        "  Chromagram:         {} bytes vs {} bytes ({:.1}x)",
        chroma_json_size.to_formatted_string(&Locale::en),
        chroma_bincode_size.to_formatted_string(&Locale::en),
        chroma_json_size as f64 / chroma_bincode_size as f64
    );
    println!(
        "  CQT:                {} bytes vs {} bytes ({:.1}x)",
        cqt_json_size.to_formatted_string(&Locale::en),
        cqt_bincode_size.to_formatted_string(&Locale::en),
        cqt_json_size as f64 / cqt_bincode_size as f64
    );

    Ok(())
}

#[cfg(feature = "serde")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_example()
}

#[cfg(not(feature = "serde"))]
fn main() {
    eprintln!("This example requires the 'serde' feature to be enabled.");
    eprintln!("Run with: cargo run --example serde_example --features serde,realfft");
    eprintln!(
        "If erroring due to missing 'num-format' dependency, make sure to run in development/debug mode. It is a dev dependency only."
    );

    std::process::exit(1);
}
