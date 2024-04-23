"""
Audio Processing for Noise Reduction

This script is designed to handle the preparation of audio datasets for training machine learning models
focused on noise reduction. This involves loading audio files, adding synthesized noise at specified levels,
converting audio into spectrogram chunks, and preparing datasets for training and validation.

Functions:
- load_audio_file(file_path, sample_rate): Loads an audio file using the librosa library, which is
  popular for audio and music analysis. The function returns the audio data along with its sample rate.

- add_noise(clean_audio, noise_audio, snr): Adds noise to a clean audio sample at a specified signal-to-noise
  ratio (SNR). This is crucial for creating training data that simulates various noisy environments.

- audio_to_spectrogram_chunked(audio, n_fft, hop_length, chunk_width, max_length_seconds, sample_rate, overlap):
  Converts audio data into spectrogram chunks using Short-Time Fourier Transform (STFT). This is useful for
  breaking down audio into manageable parts for deep learning models, which often rely on spectrograms for
  feature extraction.

- prepare_dataset(clean_files, noise_dirs, snrs, split_ratio): Orchestrates the loading of audio files,
  addition of noise, and the splitting of data into training and validation sets. It utilizes multi-threading
  to enhance processing efficiency, handling potentially large datasets by parallelizing file operations.

The script also includes an example usage section that demonstrates how to define file paths, prepare the dataset,
and save it for future use. This is particularly aimed at scenarios requiring robust models for environments
with varying noise levels, such as industrial settings or public spaces where background noise can significantly
impact audio clarity.

This modular approach not only facilitates extensive customization and testing of different audio processing
strategies but also ensures scalability and adaptability to different project requirements or research goals.

Libraries:
- os: For handling file and directory operations.
- librosa: For audio loading and transformation.
- numpy: For numerical operations on arrays.
- random: For shuffling data to ensure randomized splits.
- matplotlib.pyplot: For visualizing spectrograms for analysis and verification.
- concurrent.futures: For parallelizing data processing to speed up the dataset preparation phase.

Example Use Case:
The provided code is prepared to be executed in environments where Python and the required libraries are installed.
It demonstrates loading specified audio files, adding controlled noise, transforming these into spectrograms,
and finally preparing them into structured datasets ready for model training and validation.

"""

approach import os
import librosa
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor

def load_audio_file(file_path: str, sample_rate: int = 22050) -> tuple:
    """
    Load an audio file with librosa.
    
    Parameters:
    - file_path: Path to the audio file.
    - sample_rate: Sampling rate for loading the audio.
    
    Returns:
    - Tuple of (audio array, sample rate).
    """
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr

def add_noise(clean_audio: np.ndarray, noise_audio: np.ndarray, snr: float) -> np.ndarray:
    """
    Add noise to clean audio at a specified SNR.
    
    Parameters:
    - clean_audio: Numpy array of clean audio.
    - noise_audio: Numpy array of noise audio.
    - snr: Desired signal-to-noise ratio (in dB).
    
    Returns:
    - Noisy audio as a numpy array.
    """
    # Ensure noise_audio can be repeated if shorter than clean_audio
    if len(noise_audio) < len(clean_audio):
        repeat_count = (len(clean_audio) // len(noise_audio)) + 1
        noise_audio = np.tile(noise_audio, repeat_count)
    
    noise_audio = noise_audio[:len(clean_audio)]  # Trim to match clean_audio length

    # Calculate the Root Mean Square (RMS) of the clean audio
    rms_clean = np.sqrt(np.mean(clean_audio ** 2))
    # Calculate the RMS of the noise audio
    rms_noise = np.sqrt(np.mean(noise_audio ** 2))

    # Calculate the scaling factor for the noise to achieve the desired SNR
    scaling_factor = rms_clean / (10 ** (snr / 20))
    # Scale the noise audio
    scaled_noise = noise_audio * scaling_factor / rms_noise
    # Add the scaled noise to the clean audio
    noisy_audio = clean_audio + scaled_noise
    
    return noisy_audio

import numpy as np
import librosa

def audio_to_combined_features(audio: np.ndarray, sample_rate: int, n_fft: int = 1024, hop_length: int = 512, n_mfcc: int = 13, max_length_seconds: float = 5.0) -> list:
    """
    Extract combined spectrogram and MFCC features from audio in chunked segments.

    Parameters:
    - audio: Audio data as a numpy array.
    - sample_rate: Sampling rate of the audio.
    - n_fft: FFT window size for the spectrogram.
    - hop_length: Hop length for both STFT and MFCC.
    - n_mfcc: Number of MFCCs to extract.
    - max_length_seconds: Maximum length of audio segments to process at a time, in seconds.
    
    Returns:
    - List of combined feature chunks with a channel dimension added.
    """
    max_length_samples = int(max_length_seconds * sample_rate)
    total_samples = len(audio)
    combined_chunks = []

    for start in range(0, total_samples, max_length_samples):
        end = min(start + max_length_samples, total_samples)
        audio_segment = audio[start:end]

        # Compute the spectrogram
        S = librosa.stft(audio_segment, n_fft=n_fft, hop_length=hop_length)
        spectrogram = np.abs(S)

        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        # Normalize the spectrogram to decibels
        spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

        # Combine spectrogram and MFCCs along the frequency axis
        combined_features = np.vstack([spectrogram_db, mfccs])

        # Adding channel dimension for compatibility with CNN input requirements
        combined_features_with_channel = combined_features[..., np.newaxis]

        combined_chunks.append(combined_features_with_channel)

    return combined_chunks

def prepare_dataset(clean_files: list, noise_dirs: list, snrs: list, split_ratio: float = 0.8) -> tuple:
    """
    Prepare the dataset by loading, augmenting, and splitting into training and validation sets using combined features.
    """
    augmented_data = []

    def process_file_combinations(clean_file, noise_file):
        try:
            clean_audio, sr = load_audio_file(clean_file)
            noise_audio, _ = load_audio_file(noise_file)
            file_augmented_data = []
            for snr in snrs:
                noisy_audio = add_noise(clean_audio, noise_audio, snr)
                combined_chunks = audio_to_combined_features(noisy_audio, sr)
                clean_combined_chunks = audio_to_combined_features(clean_audio, sr)
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
        
        for future in futures:
            augmented_data.extend(future.result())

    # Shuffle the augmented data to ensure a random distribution for splitting
    random.shuffle(augmented_data)

    # Split the data into training and validation sets
    split_index = int(len(augmented_data) * split_ratio)
    train_data = augmented_data[:split_index]
    val_data = augmented_data[split_index:]

    return train_data, val_data


# Example usage
if __name__ == "__main__":
    # Define your clean audio files and noise directories
    # Clean audio file paths
    clean_files = [
        r"C:\!code\clean_audio\wav\historyamericanrevolutionvol3_01_warren_64kb.wav",
        r"C:\!code\clean_audio\wav\historyamericanrevolutionvol3_02_warren_64kb.wav",
        r"C:\!code\clean_audio\wav\historyamericanrevolutionvol3_03_warren_64kb.wav",
        r"C:\!code\clean_audio\wav\historyamericanrevolutionvol3_04_warren_64kb.wav",
        r"C:\!code\clean_audio\wav\historyamericanrevolutionvol3_05_warren_64kb.wav"
    ]

    # Noise directories from the Urbansound8k Dataset
    noise_dirs = [r"C:\!code\unclean_audio\fold" + str(i) for i in range(1, 11)]

    snrs = [0, -15, -50]  # Example SNR values

    # Prepare the dataset
    train_data, val_data = prepare_dataset(clean_files, noise_dirs, snrs)

    # Example of saving the training and validation data
    np.save('train_data.npy', train_data)
    np.save('val_data.npy', val_data)
    # ! Load the training and validation data from their respective files.
    # * This process involves using numpy's load function with 'allow_pickle=True' to handle the structured array.
    try:
        train_data_loaded = np.load('train_data.npy', allow_pickle=True)
        val_data_loaded = np.load('val_data.npy', allow_pickle=True)
    except FileNotFoundError as e:
        # If the files are not found, raise an informative error.
        raise FileNotFoundError("Training or validation data file not found. Ensure the files 'train_data.npy' and 'val_data.npy' exist in the directory.") from e
    
    # * Extract the first clean spectrogram chunk from the loaded training data.
    # This involves selecting the second element of the first tuple in the array and squeezing it to remove any singleton dimensions.
    clean_spec_chunk = train_data_loaded[0][1].squeeze()
    
    # * Plot the clean spectrogram chunk using matplotlib.
    # This visualization helps in understanding the structure and quality of the spectrogram.
    plt.figure(figsize=(10, 4))
    plt.imshow(clean_spec_chunk, aspect='auto', origin='lower')
    plt.title('Clean Spectrogram Chunk')
    plt.xlabel('Time Steps')
    plt.ylabel('Frequency Bins')
    plt.colorbar(label='Magnitude')
    plt.show()
