#!/usr/bin/env python3
"""
Chromagram Example

Demonstrates computing chromagrams (pitch class profiles) for music analysis.
Chromagrams represent the intensity of each of the 12 pitch classes (C, C#, D, ..., B)
over time, useful for chord recognition and harmonic analysis.
"""

import numpy as np
import spectrograms as sg


def generate_chord(sample_rate, duration, notes_hz, note_names):
    """Generate a chord from a list of frequencies."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float64)

    # Sum up all the notes
    signal = np.zeros_like(t)
    for freq in notes_hz:
        signal += np.sin(2 * np.pi * freq * t)

    # Normalize
    signal /= len(notes_hz)

    # Apply envelope to make it more realistic
    envelope = np.exp(-3 * t / duration)  # Decay
    signal *= envelope

    return signal


def main():
    print("=" * 60)
    print("Chromagram (Pitch Class Profile) Example")
    print("=" * 60)

    sample_rate = 22050  # Higher sample rate for music
    duration = 2.0

    # ========================================================================
    # Generate musical test signals
    # ========================================================================
    print("\n" + "=" * 60)
    print("Generating Musical Test Signals")
    print("=" * 60)

    # Musical note frequencies (Hz)
    # A4 = 440 Hz, and we'll use equal temperament
    A4 = 440.0

    # Generate a C major chord (C4, E4, G4)
    C4 = A4 * 2 ** (-9 / 12)  # C4 = 261.63 Hz
    E4 = A4 * 2 ** (-5 / 12)  # E4 = 329.63 Hz
    G4 = A4 * 2 ** (-2 / 12)  # G4 = 392.00 Hz

    c_major = generate_chord(sample_rate, duration, [C4, E4, G4], "C major")

    print(f"\nC major chord (C-E-G):")
    print(f"  C4: {C4:.2f} Hz")
    print(f"  E4: {E4:.2f} Hz")
    print(f"  G4: {G4:.2f} Hz")

    # Generate an A minor chord (A4, C5, E5)
    A4_note = A4
    C5 = A4 * 2 ** (3 / 12)  # C5 = 523.25 Hz
    E5 = A4 * 2 ** (7 / 12)  # E5 = 659.25 Hz

    a_minor = generate_chord(sample_rate, duration, [A4_note, C5, E5], "A minor")

    print(f"\nA minor chord (A-C-E):")
    print(f"  A4: {A4_note:.2f} Hz")
    print(f"  C5: {C5:.2f} Hz")
    print(f"  E5: {E5:.2f} Hz")

    # ========================================================================
    # Compute chromagram with standard parameters
    # ========================================================================
    print("\n" + "=" * 60)
    print("Computing Chromagrams")
    print("=" * 60)

    # STFT parameters (larger FFT for better frequency resolution)
    stft = sg.StftParams(
        n_fft=4096,  # Larger FFT for music
        hop_size=512,  # ~23ms hop
        window="hanning",
        centre=True,
    )

    print(f"\nSTFT parameters:")
    print(f"  FFT size: {stft.n_fft} ({stft.n_fft / sample_rate * 1000:.1f} ms)")
    print(f"  Hop size: {stft.hop_size} ({stft.hop_size / sample_rate * 1000:.1f} ms)")

    # Use standard music parameters
    chroma_params = sg.ChromaParams.music_standard()

    print(f"\nChroma parameters:")
    print(f"  Tuning: {chroma_params.tuning} Hz (A4)")
    print(f"  Frequency range: {chroma_params.f_min} - {chroma_params.f_max} Hz")

    # Compute chromagrams for both chords
    print("\nComputing chromagram for C major chord...")
    chroma_c_major = sg.compute_chromagram(c_major, stft, sample_rate, chroma_params)

    print("C major chromagram computed")
    print(
        f"  Shape: {chroma_c_major.shape} (12 pitch classes x {chroma_c_major.shape[1]} frames)"
    )

    print("\nComputing chromagram for A minor chord...")
    chroma_a_minor = sg.compute_chromagram(a_minor, stft, sample_rate, chroma_params)

    print("A minor chromagram computed")
    print(
        f"  Shape: {chroma_a_minor.shape} (12 pitch classes x {chroma_a_minor.shape[1]} frames)"
    )

    # ========================================================================
    # Analyze chromagram content
    # ========================================================================
    print("\n" + "=" * 60)
    print("Chromagram Analysis")
    print("=" * 60)

    # Pitch class names
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Average over time for each chord
    c_major_avg = np.mean(chroma_c_major, axis=1)
    a_minor_avg = np.mean(chroma_a_minor, axis=1)

    # Normalize to [0, 1]
    c_major_avg /= np.max(c_major_avg)
    a_minor_avg /= np.max(a_minor_avg)

    print("\nC Major Chord - Average Chroma Profile:")
    print(f"{'Pitch Class':>12}  {'Energy':>8}  {'Bar':>20}")
    print("-" * 50)

    for i, (pc, energy) in enumerate(zip(pitch_classes, c_major_avg)):
        bar = "█" * int(energy * 20)
        marker = " ← Expected" if pc in ["C", "E", "G"] else ""
        print(f"{pc:>12}  {energy:>8.3f}  {bar:<20}{marker}")

    print("\nA Minor Chord - Average Chroma Profile:")
    print(f"{'Pitch Class':>12}  {'Energy':>8}  {'Bar':>20}")
    print("-" * 50)

    for i, (pc, energy) in enumerate(zip(pitch_classes, a_minor_avg)):
        bar = "█" * int(energy * 20)
        marker = " ← Expected" if pc in ["A", "C", "E"] else ""
        print(f"{pc:>12}  {energy:>8.3f}  {bar:<20}{marker}")

    # ========================================================================
    # Chord recognition
    # ========================================================================
    print("\n" + "=" * 60)
    print("Simple Chord Recognition")
    print("=" * 60)

    # Define chord templates (which pitch classes are in each chord)
    chord_templates = {
        "C major": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C, E, G
        "A minor": [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # A, C, E
        "G major": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # G, B, D (B=11, D=2)
        "F major": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # F, A, C
    }
    # Fix G major template
    chord_templates["G major"] = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]  # D, G, B

    def recognize_chord(chroma_profile, templates):
        """Recognize chord using template matching."""
        chroma_normalized = chroma_profile / (np.linalg.norm(chroma_profile) + 1e-10)

        best_match = None
        best_score = -1

        for chord_name, template in templates.items():
            template_array = np.array(template, dtype=float)
            template_normalized = template_array / (
                np.linalg.norm(template_array) + 1e-10
            )

            # Cosine similarity
            score = np.dot(chroma_normalized, template_normalized)

            if score > best_score:
                best_score = score
                best_match = chord_name

        return best_match, best_score

    # Recognize the chords
    c_major_recognized, c_major_score = recognize_chord(c_major_avg, chord_templates)
    a_minor_recognized, a_minor_score = recognize_chord(a_minor_avg, chord_templates)

    print("\nChord Recognition Results:")
    print(f"\nSignal 1 (C major chord):")
    print(f"  Recognized as: {c_major_recognized}")
    print(f"  Confidence: {c_major_score:.3f}")

    print(f"\nSignal 2 (A minor chord):")
    print(f"  Recognized as: {a_minor_recognized}")
    print(f"  Confidence: {a_minor_score:.3f}")

    # ========================================================================
    # Time evolution of chroma features
    # ========================================================================
    print("\n" + "=" * 60)
    print("Time Evolution")
    print("=" * 60)

    # Show how chroma features evolve over time (first few frames)
    n_show = min(5, chroma_c_major.shape[1])

    print(f"\nC major chord - First {n_show} frames:")
    print(f"{'Frame':>6}  " + "  ".join(f"{pc:>5}" for pc in pitch_classes))
    print("-" * 80)

    for frame_idx in range(n_show):
        values = chroma_c_major[:, frame_idx]
        values_normalized = values / (np.max(values) + 1e-10)
        print(f"{frame_idx:>6}  " + "  ".join(f"{v:>5.2f}" for v in values_normalized))

    # ========================================================================
    # Applications and tips
    # ========================================================================
    print("\n" + "=" * 60)
    print("Applications and Guidelines")
    print("=" * 60)

    print("\nCommon applications:")
    print("  1. Chord recognition")
    print("     • Match chroma profiles to chord templates")
    print("     • Useful for automatic transcription")

    print("\n  2. Key detection")
    print("     • Aggregate chroma over longer time windows")
    print("     • Match to major/minor key profiles")

    print("\n  3. Cover song identification")
    print("     • Chroma features are robust to instrumentation")
    print("     • Compare chroma sequences between recordings")

    print("\n  4. Harmonic similarity")
    print("     • Measure similarity between musical pieces")
    print("     • Useful for music recommendation")

    print("\nParameter selection:")
    print("  • Large FFT size (2048-8192) for better frequency resolution")
    print("  • Smaller hop size for better time resolution")
    print("  • Use standard tuning (440 Hz) unless analyzing historical recordings")

    print("\nNormalization:")
    print("  • L2 norm (default): Good for chord recognition")
    print("  • L1 norm: Alternative for some applications")
    print("  • Max norm: Emphasizes strongest pitch class")

    print("\nChromagram example completed!")


if __name__ == "__main__":
    main()
