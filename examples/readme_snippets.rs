use non_empty_slice::NonEmptyVec;
use spectrograms::*;
use std::f64::consts::PI;

fn samples() -> NonEmptyVec<f64> {
    let sample_rate = 16000.0;
    let samples: Vec<f64> = (0..16000)
        .map(|i| {
            let t = i as f64 / sample_rate;
            (2.0 * PI * 440.0 * t).sin()
        })
        .collect();
    let samples = NonEmptyVec::new(samples).unwrap();
    samples
}

pub fn main() -> Result<(), SpectrogramError> {
    println!("---");
    println!("Basic spectrogram example:");
    println!("---");
    basic_spectrogram_example()?;

    println!("---");
    println!("Mel spectrogram example:");
    println!("---");

    mel_spectrogram_example()?;

    println!("---");
    println!("Efficient batch processing example:");
    println!("---");

    efficient_batch_processing()?;

    println!("---");
    println!("2D FFT and image processing example:");
    println!("---");

    _2d_fft_and_image_processing()?;

    println!("---");
    println!("Efficient batch image processing example:");
    println!("---");

    efficient_batch_image_processing()?;

    println!("---");
    println!("MFCC example:");
    println!("---");

    mfccs()?;

    println!("---");
    println!("Chromagram example:");
    println!("---");

    _chromagram()?;
    Ok(())
}

fn basic_spectrogram_example() -> SpectrogramResult<()> {
    let samples = samples();
    // Configure parameters
    let stft = StftParams::new(
        nzu!(512),           // FFT size
        nzu!(256),           // hop size
        WindowType::Hanning, // window
        true,                // centre frames
    )?;

    let params = SpectrogramParams::new(
        stft, 16000.0, // sample rate
    )?;

    // Compute power spectrogram
    let spec = LinearPowerSpectrogram::compute(samples.as_non_empty_slice(), &params, None)?;

    println!("Shape: {} bins x {} frames", spec.n_bins(), spec.n_frames());
    Ok(())
}

fn mel_spectrogram_example() -> SpectrogramResult<()> {
    let samples = samples();
    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    let params = SpectrogramParams::new(stft, 16000.0)?;

    // Mel filterbank
    let mel = MelParams::new(
        nzu!(80), // n_mels
        0.0,      // f_min
        8000.0,   // f_max
    )?;

    // dB scaling
    let db = LogParams::new(-80.0)?;

    // Compute mel spectrogram in dB
    let spec = MelDbSpectrogram::compute(&samples, &params, &mel, Some(&db))?;

    // Access data
    println!("Mel bands: {}", spec.n_bins());
    println!("Frames: {}", spec.n_frames());
    println!("Frequency range: {:?}", spec.axes().frequency_range());

    Ok(())
}

fn efficient_batch_processing() -> SpectrogramResult<()> {
    use non_empty_slice::non_empty_vec;
    use spectrograms::*;
    let signals = vec![
        non_empty_vec![0.0; nzu!(16000)],
        non_empty_vec![0.0; nzu!(16000)],
        non_empty_vec![0.0; nzu!(16000)],
    ];

    let stft = StftParams::new(nzu!(512), nzu!(256), WindowType::Hanning, true)?;
    let params = SpectrogramParams::new(stft, 16000.0)?;
    let mel = MelParams::new(nzu!(80), 0.0, 8000.0)?;
    let db = LogParams::new(-80.0)?;

    // Create plan once
    let planner = SpectrogramPlanner::new();
    let mut plan = planner.mel_plan::<Decibels>(&params, &mel, Some(&db))?;

    // Reuse for all signals (much faster!)
    for signal in signals {
        let _spec = plan.compute(&signal)?;
        // Process spec...
    }

    Ok(())
}

fn _2d_fft_and_image_processing() -> SpectrogramResult<()> {
    use ndarray::Array2;
    use spectrograms::fft2d::*;
    use spectrograms::image_ops::*;

    // Create a 256x256 image
    let image = Array2::<f64>::from_shape_fn((256, 256), |(i, j)| {
        ((i as f64 - 128.0).powi(2) + (j as f64 - 128.0).powi(2)).sqrt()
    });

    // Compute 2D FFT
    let spectrum = fft2d(&image.view())?;
    println!("Spectrum shape: {:?}", spectrum.shape());
    // Output: [256, 129] due to Hermitian symmetry

    // Apply Gaussian blur via FFT
    let kernel = gaussian_kernel_2d(spectrograms::nzu!(9), 2.0)?;

    let _blurred = convolve_fft(&image.view(), &kernel.view())?;

    // Apply high-pass filter for edge detection
    let _edges = highpass_filter(&image.view(), 0.1)?;

    // Compute power spectrum
    let _power = power_spectrum_2d(&image.view())?;

    Ok(())
}

fn efficient_batch_image_processing() -> SpectrogramResult<()> {
    use ndarray::Array2;
    use spectrograms::fft2d::Fft2dPlanner;

    let images = vec![
        Array2::<f64>::zeros((256, 256)),
        Array2::<f64>::zeros((256, 256)),
        Array2::<f64>::zeros((256, 256)),
    ];

    // Create planner once
    let mut planner = Fft2dPlanner::new();

    // Reuse for all images (faster!)
    for image in &images {
        let spectrum = planner.fft2d(&image.view())?;
        let _power = spectrum.mapv(|c| c.norm_sqr());
        // Process power spectrum...
    }

    Ok(())
}

fn mfccs() -> SpectrogramResult<()> {
    let samples = samples();
    let stft = StftParams::new(nzu!(512), nzu!(160), WindowType::Hanning, true)?;
    let mfcc_params = MfccParams::new(nzu!(13));

    let mfccs = mfcc(
        &samples,
        &stft,
        16000.0,
        nzu!(40), // n_mels
        &mfcc_params,
    )?;

    // Shape: (13, n_frames)
    println!("MFCCs: {} x {}", mfccs.nrows(), mfccs.ncols());
    Ok(())
}

fn _chromagram() -> SpectrogramResult<()> {
    let samples = samples();
    let stft = StftParams::new(nzu!(4096), nzu!(512), WindowType::Hanning, true)?;
    let chroma_params = ChromaParams::music_standard();

    let chroma = chromagram(&samples, &stft, 22050.0, &chroma_params)?;

    // Shape: (12, n_frames) - one row per pitch class
    println!("Chroma: {} x {}", chroma.nrows(), chroma.ncols());

    Ok(())
}

#[allow(unused)]
fn window_functions() -> SpectrogramResult<()> {
    // Parse from string
    let window: WindowType = "hanning".parse()?;
    let kaiser: WindowType = "kaiser=8.0".parse()?;

    // Or use constructors
    let hann = WindowType::Hanning;
    let gauss = WindowType::Gaussian { std: 0.4 };

    // Generate windows
    let hann_window = make_window(WindowType::Hanning, nzu!(512));
    let kaiser_window = make_window(WindowType::Kaiser { beta: 8.0 }, nzu!(512));
    // etc.

    Ok(())
}

#[allow(unused)]
fn default_presets() -> SpectrogramResult<()> {
    // Speech processing preset
    // n_fft=512, hop_size=160
    let params = SpectrogramParams::speech_default(16000.0)?;

    // Music processing preset
    // n_fft=2048, hop_size=512
    let params = SpectrogramParams::music_default(44100.0)?;

    Ok(())
}

#[allow(unused)]
fn accessing_results() -> SpectrogramResult<()> {
    let samples = samples();
    let params = SpectrogramParams::speech_default(16000.0)?;
    let spec = LinearPowerSpectrogram::compute(&samples, &params, None)?;

    // Dimensions
    let n_bins = spec.n_bins();
    let n_frames = spec.n_frames();

    // Data (ndarray::Array2<f64>)
    let data = spec.data();

    // Axes
    let freqs = spec.axes().frequencies();
    let times = spec.axes().times();
    let (f_min, f_max) = spec.axes().frequency_range();
    let duration = spec.axes().duration();

    // Original parameters
    let params = spec.params();

    Ok(())
}