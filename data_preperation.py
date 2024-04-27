"""
data_preparation.py
Audio Processing for Noise Reduction

This Python script is dedicated to the preparation and handling of audio datasets for training machine learning models focused on noise reduction. The script includes functions for loading audio files, synthesizing noise at specified signal-to-noise ratios, and extracting combined audio features (spectrograms and Mel Frequency Cepstral Coefficients, MFCCs). These combined features are crucial for training robust models capable of distinguishing speech from background noise in complex acoustic environments.

The script is structured to perform the following tasks:
- Load audio data from specified file paths using the librosa library.
- Synthesize and add noise to clean audio samples to simulate various noisy conditions.
- Extract combined features (spectrograms and MFCCs) from audio data in chunked segments.
- Organize audio data by loading, augmenting, and then splitting it into training and validation sets.
- Save processed data to disk for persistent access and future training sessions.
- Visually inspect the extracted features using matplotlib.

Libraries Used:
- os: For directory and file operations.
- librosa: For loading and transforming audio files.
- numpy: For high-performance numerical operations.
- random: For shuffling data.
- matplotlib.pyplot: For visualizing data.
- concurrent.futures: For parallel processing to expedite the preparation of the dataset.

Example Usage:
The script is executed within an environment where Python and all required libraries are installed. It demonstrates the complete workflow from defining file paths, processing audio files to extract features, preparing the dataset, and finally visualizing the prepared data. This is aimed at scenarios requiring robust noise reduction models for environments like industrial settings or urban areas, where background noise can significantly impact the clarity of audio communications.
"""

import os
import librosa
import numpy as np
import random
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to load an audio file
def load_audio_file(file_path: str, sample_rate: int = 22050) -> tuple:
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr

# Function to add noise to an audio file
def add_noise(clean_audio: np.ndarray, noise_audio: np.ndarray, snr: float) -> np.ndarray:
    rms_clean = np.sqrt(np.mean(clean_audio**2))
    rms_noise = np.sqrt(np.mean(noise_audio**2))
    scaling_factor = rms_clean / (10**(snr / 20)) / rms_noise
    noisy_audio = clean_audio + noise_audio * scaling_factor
    return noisy_audio

# Function to extract combined audio features
def audio_to_combined_features_stream(file_path: str, sample_rate: int = 22050, n_fft: int = 1024, hop_length: int = 512, n_mfcc: int = 13, max_length_seconds: float = 5.0) -> list:
    max_length_samples = int(max_length_seconds * sample_rate)
    stream = librosa.stream(file_path, block_length=int(max_length_seconds), frame_length=n_fft, hop_length=hop_length, mono=True, sr=sample_rate)
    combined_chunks = []
    for audio_segment in stream:
        if len(audio_segment) < max_length_samples:
            continue
        S = librosa.stft(audio_segment, n_fft=n_fft, hop_length=hop_length)
        spectrogram = np.abs(S)
        mfccs = librosa.feature.mfcc(S=spectrogram, sr=sample_rate, n_mfcc=n_mfcc)
        spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
        combined_features = np.vstack([spectrogram_db, mfccs])
        combined_features_with_channel = combined_features[..., np.newaxis]
        combined_chunks.append(combined_features_with_channel)
    return combined_chunks

# Main function to prepare the dataset
def prepare_dataset(clean_files: list, noise_dirs: list, snrs: list, split_ratio: float = 0.8) -> tuple:
    augmented_data = []
    def process_file_combinations(clean_file, noise_file):
        try:
            print(f"Processing: {clean_file} with noise {noise_file}")
            clean_audio, sr = load_audio_file(clean_file)
            noise_audio, _ = load_audio_file(noise_file)
            file_augmented_data = []
            for snr in snrs:
                print(f"Adding noise at SNR: {snr} dB")
                noisy_audio = add_noise(clean_audio, noise_audio, snr)
                combined_chunks = audio_to_combined_features_stream(clean_file, sr)
                clean_combined_chunks = audio_to_combined_features_stream(clean_file, sr)
                file_augmented_data.extend(list(zip(combined_chunks, clean_combined_chunks)))
            return file_augmented_data
        except Exception as e:
            print(f"Failed processing {clean_file} or {noise_file}: {str(e)}")
            return []

    with ThreadPoolExecutor() as executor:
        futures = []
        for clean_file in clean_files:
            for noise_dir in noise_dirs:
                noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')]
                for noise_file in noise_files:
                    futures.append(executor.submit(process_file_combinations, clean_file, noise_file))

        for future in as_completed(futures):
            result = future.result()
            if result:
                augmented_data.extend(result)
                print(f"Processed {len(result)} combinations")

    print("Shuffling and splitting the dataset...")
    random.shuffle(augmented_data)
    split_index = int(len(augmented_data) * split_ratio)
    train_data = augmented_data[:split_index]
    val_data = augmented_data[split_index:]
    print("Dataset preparation complete.")
    return train_data, val_data

# Example usage
if __name__ == "__main__":
    clean_files = [
        r"C:\!code\clean_audio\wav\historyamericanrevolutionvol3_01_warren_64kb.wav",
        r"C:\!code\clean_audio\wav\historyamericanrevolutionvol3_02_warren_64kb.wav",
        r"C:\!code\clean_audio\wav\historyamericanrevolutionvol3_03_warren_64kb.wav",
        r"C:\!code\clean_audio\wav\historyamericanrevolutionvol3_04_warren_64kb.wav",
        r"C:\!code\clean_audio\wav\historyamericanrevolutionvol3_05_warren_64kb.wav"
    ]
    noise_dirs = [r"C:\!code\unclean_audio\fold" + str(i) for i in range(1, 11)]
    snrs = [0, -15, -50]
  
    train_data, val_data = prepare_dataset(clean_files, noise_dirs, snrs)
    np.save('train_data.npy', train_data)
    np.save('val_data.npy', val_data)
  
    try:
        train_data_loaded = np.load('train_data.npy', allow_pickle=True)
        val_data_loaded = np.load('val_data.npy', allow_pickle=True)
    except FileNotFoundError as e:
        print("File not found error:", e)
  
    combined_features_chunk = train_data_loaded[0][1].squeeze()
    plt.figure(figsize=(10, 4))
    plt.imshow(combined_features_chunk, aspect='auto', origin='lower')
    plt.title('Combined Spectrogram and MFCC Chunk')
    plt.xlabel('Time Steps')
    plt.ylabel('Frequency Bins')
    plt.colorbar(label='Magnitude')
    plt.show()
