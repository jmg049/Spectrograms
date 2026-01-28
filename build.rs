fn main() {
    #[cfg(feature = "fftw")]
    {
        println!("cargo:rerun-if-changed=build.rs");

        // Try to find fftw3 using pkg-config (standard on Linux/macOS)
        if let Err(e) = pkg_config::Config::new()
            .atleast_version("3.3")
            .probe("fftw3")
        {
            // This prints a readable warning in the build logs
            println!("cargo:warning=FFTW3 development files not found: {}", e);
            println!(
                "cargo:warning=If compilation fails, install libfftw3-dev (Linux) or fftw (Homebrew)."
            );
        } else {
            println!("cargo:rustc-cfg=fftw_found");
        }
    }
}
