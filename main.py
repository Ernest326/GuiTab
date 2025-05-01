import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import librosa
from scipy.signal import find_peaks

SAMPLE_RATE = 44100
FRAME_SIZE = 2048

def get_note_name(frequency):
    A4 = 440.0 # Standard tuning for A4
    C0 = A4 * (2 ** (-4.75))  # Calculate C0 frequency and use as base note
    
    # How many semitones from C0 to the note
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Calculate the note number
    note_number = int(round(12 * np.log2(frequency / C0)))
    
    # Get the note name and octave
    note_name = note_names[note_number % 12]
    octave = note_number // 12
    
    return f"{note_name}{octave}"


def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    
    #TODO: Some noise filtering could be applied before FFT
    
    # Apply window to audio frame and compute FFT
    frame = indata[:, 0] * np.hanning(len(indata)) # Apply a Hanning window to the audio frame
    spectrum = np.fft.fft(frame)
    freqs = np.fft.fftfreq(len(spectrum), d=1/SAMPLE_RATE)
    magnitudes = np.abs(spectrum)

    
    # Remove negative frequencies and corresponding magnitudes
    freqs = freqs[:len(freqs)//2]
    magnitudes = magnitudes[:len(magnitudes)//2]
    
    # Find peaks in the magnitude spectrum
    peaks, _ = find_peaks(magnitudes, height=0.1) # TODO: Adjust height threshold based input
    peak_freqs = freqs[peaks]
    peak_mags = magnitudes[peaks]
    
    # Sort peaks by mag
    sorted_peaks = sorted(zip(peak_mags, peak_freqs), reverse=True)
    
    # Take the top 5 peaks
    top_peaks = sorted_peaks[:5]
    notes = [get_note_name(freq) for _, freq in top_peaks]
    
    print("Detected notes:", notes)
    
    plt.clf()
    plt.plot(freqs, magnitudes)
    plt.title("FFT Magnitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, SAMPLE_RATE // 2)
    plt.ylim(0, np.max(magnitudes) * 1.1)
    plt.grid()
    plt.pause(0.01)
    

if __name__ == "__main__":
    
    # Set the audio input device
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']}")
    device_index = int(input("-> :"))
    sd.default.device = device_index
    
    # Start audio stream callback
    with sd.InputStream(callback=audio_callback, samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE):
        input("Press Enter to stop the stream...")

#TODO: Implement separate methods, classes etc.
# Maybe also use multithreading for audio processing and plotting