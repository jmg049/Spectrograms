use spectrograms::WindowType;
use std::str::FromStr;

#[test]
fn test_window_from_str_rectangular() {
    assert_eq!(
        WindowType::from_str("rectangle").unwrap(),
        WindowType::Rectangular
    );
    assert_eq!(
        WindowType::from_str("rect").unwrap(),
        WindowType::Rectangular
    );
    assert_eq!(
        WindowType::from_str("RECT").unwrap(),
        WindowType::Rectangular
    );
}

#[test]
fn test_window_from_str_hanning() {
    assert_eq!(
        WindowType::from_str("hanning").unwrap(),
        WindowType::Hanning
    );
    assert_eq!(WindowType::from_str("hann").unwrap(), WindowType::Hanning);
    assert_eq!(WindowType::from_str("HANN").unwrap(), WindowType::Hanning);
}

#[test]
fn test_window_from_str_hamming() {
    assert_eq!(
        WindowType::from_str("hamming").unwrap(),
        WindowType::Hamming
    );
    assert_eq!(WindowType::from_str("hamm").unwrap(), WindowType::Hamming);
}

#[test]
fn test_window_from_str_blackman() {
    assert_eq!(
        WindowType::from_str("blackman").unwrap(),
        WindowType::Blackman
    );
    assert_eq!(
        WindowType::from_str("BLACKMAN").unwrap(),
        WindowType::Blackman
    );
}

#[test]
fn test_window_from_str_kaiser() {
    let kaiser = WindowType::from_str("kaiser=5.0").unwrap();
    match kaiser {
        WindowType::Kaiser { beta } => {
            assert!((beta - 5.0).abs() < 1e-10);
        }
        _ => panic!("Expected Kaiser window"),
    }

    let kaiser2 = WindowType::from_str("KAISER=10.5").unwrap();
    match kaiser2 {
        WindowType::Kaiser { beta } => {
            assert!((beta - 10.5).abs() < 1e-10);
        }
        _ => panic!("Expected Kaiser window"),
    }
}

#[test]
fn test_window_from_str_gaussian() {
    let gaussian = WindowType::from_str("gaussian=2.5").unwrap();
    match gaussian {
        WindowType::Gaussian { std } => {
            assert!((std - 2.5).abs() < 1e-10);
        }
        _ => panic!("Expected Gaussian window"),
    }
}

#[test]
fn test_window_from_str_invalid() {
    assert!(WindowType::from_str("").is_err());
    assert!(WindowType::from_str("invalid").is_err());
    assert!(WindowType::from_str("kaiser").is_err()); // Missing parameter
    assert!(WindowType::from_str("gaussian").is_err()); // Missing parameter
    assert!(WindowType::from_str("kaiser=").is_err()); // Empty parameter
    assert!(WindowType::from_str("kaiser=abc").is_err()); // Invalid number
}

#[test]
fn test_window_display() {
    assert_eq!(WindowType::Rectangular.to_string(), "Rectangular");
    assert_eq!(WindowType::Hanning.to_string(), "Hanning");
    assert_eq!(WindowType::Hamming.to_string(), "Hamming");
    assert_eq!(WindowType::Blackman.to_string(), "Blackman");
    assert_eq!(
        WindowType::Kaiser { beta: 5.0 }.to_string(),
        "Kaiser(beta=5)"
    );
    assert_eq!(
        WindowType::Gaussian { std: 2.5 }.to_string(),
        "Gaussian(std=2.5)"
    );
}

#[test]
fn test_window_default() {
    assert_eq!(WindowType::default(), WindowType::Hanning);
}

#[test]
fn test_window_clone_eq() {
    let w1 = WindowType::Hanning;
    let w2 = w1.clone();
    assert_eq!(w1, w2);

    let w3 = WindowType::Kaiser { beta: 5.0 };
    let w4 = w3.clone();
    assert_eq!(w3, w4);
}
