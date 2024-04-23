"""
Audio Processing for Noise Reduction

This Python script is dedicated to the preparation and handling of audio datasets for training machine learning models focused on noise reduction. The script includes functions for loading audio files, synthesizing noise at specified signal-to-noise ratios, and extracting combined audio features (spectrograms and Mel Frequency Cepstral Coefficients, MFCCs). These combined features are crucial for training robust models capable of distinguishing speech from background noise in complex acoustic environments.

The script is structured to perform the following tasks:
- Load audio data from specified file paths using the librosa library, which is widely recognized for its powerful audio processing capabilities.
- Synthesize and add noise to clean audio samples to simulate various noisy conditions, essential for creating realistic training scenarios.
- Extract combined features (spectrograms and MFCCs) from audio data. This is performed in chunked segments to manage memory usage effectively and to prepare the data for input into convolutional neural networks (CNNs), which benefit from this combined feature approach.
- Organize audio data by loading, augmenting, and then splitting it into training and validation sets to ensure thorough evaluation and validation of the machine learning model. This process utilizes multi-threading to enhance efficiency, particularly useful when handling large datasets.
- Save processed data to disk for persistent access and future training sessions.
- Visually inspect the extracted features using matplotlib to ensure data integrity and to understand the characteristics of the processed audio.

Libraries Used:
- os: For directory and file operations.
- librosa: For loading and transforming audio files.
- numpy: For high-performance numerical operations.
- random: For shuffling data to ensure a randomized split between training and validation sets.
- matplotlib.pyplot: For visualizing data, crucial for verifying the correct processing of audio features.
- concurrent.futures: For parallel processing to expedite the preparation of the dataset.

Example Usage:
The script is executed within an environment where Python and all required libraries are installed. It demonstrates the complete workflow from defining file paths, processing audio files to extract features, preparing the dataset, and finally visualizing the prepared data. This is particularly aimed at scenarios requiring robust noise reduction models for environments like industrial settings or urban areas, where background noise can significantly impact the clarity of audio communications.

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
  clean_files = [
      r"C:\!code\clean_audio\wav\historyamericanrevolutionvol3_01_warren_64kb.wav",
      r"C:\!code\clean_audio\wav\historyamericanrevolutionvol3_02_warren_64kb.wav",
      r"C:\!code\clean_audio\wav\historyamericanrevolutionvol3_03_warren_64kb.wav",
      r"C:\!code\clean_audio\wav\historyamericanrevolutionvol3_04_warren_64kb.wav",
      r"C:\!code\clean_audio\wav\historyamericanrevolutionvol3_05_warren_64kb.wav"
  ]
  
  # Noise directories from the Urbansound8k Dataset
  noise_dirs = [r"C:\!code\unclean_audio\fold" + str(i) for i in range(1, 11)]
  
  # Example SNR values
  snrs = [0, -15, -50]
  
  # Prepare the dataset
  train_data, val_data = prepare_dataset(clean_files, noise_dirs, snrs)
  
  # Example of saving the training and validation data
  np.save('train_data.npy', train_data)
  np.save('val_data.npy', val_data)
  
  # Load the training and validation data from their respective files.
  try:
      train_data_loaded = np.load('train_data.npy', allow_pickle=True)
      val_data_loaded = np.load('val_data.npy', allow_pickle=True)
  except FileNotFoundError as e:
      # If the files are not found, raise an informative error.
      raise FileNotFoundError("Training or validation data file not found. Ensure the files 'train_data.npy' and 'val_data.npy' exist in the directory.") from e
  
  # Extract the first combined feature chunk (spectrogram and MFCCs) from the loaded training data.
  # This involves selecting the second element of the first tuple in the array and squeezing it to remove any singleton dimensions.
  combined_features_chunk = train_data_loaded[0][1].squeeze()
  
  # Plot the combined feature chunk using matplotlib.
  # This visualization helps in understanding the structure and quality of the spectrogram and MFCC combined.
  plt.figure(figsize=(10, 4))
  plt.imshow(combined_features_chunk, aspect='auto', origin='lower')
  plt.title('Combined Spectrogram and MFCC Chunk')
  plt.xlabel('Time Steps')
  plt.ylabel('Frequency Bins')
  plt.colorbar(label='Magnitude')
  plt.show()

