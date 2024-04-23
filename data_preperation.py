approach import os
import librosa
import numpy as np
import random
import matplotlib.pyplot as plt

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
    # Ensure noise_audio is the same length as clean_audio
    min_length = min(len(clean_audio), len(noise_audio))
    clean_audio = clean_audio[:min_length]
    noise_audio = noise_audio[:min_length]

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

def audio_to_spectrogram_chunked(audio: np.ndarray, n_fft: int = 1024, hop_length: int = 1024, chunk_width: int = 173, max_length_seconds: float = 5.0, sample_rate: int = 22050) -> list:
    """
    Convert an audio waveform to a series of spectrogram chunks, processing the audio in smaller segments to reduce memory usage.
    
    Parameters:
    - audio: Audio data as a numpy array.
    - n_fft: FFT window size.
    - hop_length: Hop length for STFT.
    - chunk_width: Desired width of each spectrogram chunk.
    - max_length_seconds: Maximum length of audio segments to process at a time, in seconds.
    - sample_rate: Sampling rate of the audio.
    
    Returns:
    - List of spectrogram chunks with a channel dimension added.
    """
    max_length_samples = int(max_length_seconds * sample_rate)
    total_samples = len(audio)
    chunks_with_channel = []

    for start in range(0, total_samples, max_length_samples):
        end = min(start + max_length_samples, total_samples)
        audio_segment = audio[start:end]
        
        # Convert segment to float16 to reduce memory usage
        audio_segment = audio_segment.astype(np.float16)
        
        S = librosa.stft(audio_segment, n_fft=n_fft, hop_length=hop_length)
        spectrogram_segment = np.abs(S).astype(np.float32)  # Convert back to float32 for further processing
        
        # Calculate the number of chunks for this segment
        num_chunks = max(1, spectrogram_segment.shape[1] // chunk_width)
        
        # Split the spectrogram segment into chunks
        segment_chunks = np.array_split(spectrogram_segment, num_chunks, axis=1)
        
        # Add a channel dimension to each chunk
        segment_chunks_with_channel = [chunk[..., np.newaxis] for chunk in segment_chunks]
        chunks_with_channel.extend(segment_chunks_with_channel)
    
    return chunks_with_channel

def prepare_dataset(clean_files: list, noise_dirs: list, snrs: list, split_ratio: float = 0.8) -> tuple:
    """
    Prepare the dataset by loading, augmenting, and splitting into training and validation sets.
    
    Parameters:
    - clean_files: List of paths to clean audio files.
    - noise_dirs: List of directories containing noise audio files.
    - snrs: List of SNR levels to use for noise addition.
    - split_ratio: Ratio to split the dataset into training and validation sets.
    
    Returns:
    - Tuple of (train_data, val_data).
    """
    augmented_data = []

    for clean_file in clean_files:
        clean_audio, sr = load_audio_file(clean_file)
        for noise_dir in noise_dirs:
            noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')]
            for noise_file in noise_files:
                noise_audio, _ = load_audio_file(noise_file)
                for snr in snrs:
                    noisy_audio = add_noise(clean_audio, noise_audio, snr)
                    # Update the function call here to use the chunked version
                    noisy_chunks = audio_to_spectrogram_chunked(noisy_audio)
                    clean_chunks = audio_to_spectrogram_chunked(clean_audio)
                    augmented_data.extend(list(zip(noisy_chunks, clean_chunks)))

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
