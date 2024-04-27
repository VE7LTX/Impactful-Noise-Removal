"""
data_augmentation.py
Audio Augmentation for Noise Reduction Model Training

This Python script complements 'data_preparation.py' by further processing the audio data through various augmentation techniques. These augmentations aim to enhance the robustness of machine learning models by introducing a variety of realistic and challenging noise scenarios. This is crucial for training models to effectively reduce noise in diverse and unpredictable environments.

The script performs the following tasks:
- Apply advanced audio transformations such as adding synthetic noise, changing pitch, and time-stretching to simulate a wide range of acoustic conditions.
- Each augmentation technique is designed to prepare the audio data for more complex noise environments that models may encounter, improving their ability to generalize from training to real-world application.
- Save the augmented audio files to disk, enabling persistent access for model training and evaluation.
- Utilize multiprocessing to handle large datasets efficiently, ensuring that the augmentation process can scale with the needs of large-scale industrial or urban applications.

Libraries Used:
- librosa: For advanced audio transformations.
- numpy: For handling numerical operations.
- soundfile: For reading and writing audio files.
- os: For directory and file operations.
- multiprocessing: For leveraging multiple CPU cores to speed up the processing of large audio datasets.

Example Usage:
The script is structured to be executed in an environment where Python and all required libraries are installed. It is designed to seamlessly integrate with 'data_preparation.py', following the preparation and initial handling of audio datasets. This script specifically targets the augmentation phase, applying various transformations and saving the processed outputs for subsequent model training phases. It is especially suited for scenarios requiring robust noise reduction capabilities, such as in industrial settings or densely populated urban areas where background noise levels are high.

Configuration and execution details are provided to customize the augmentation process based on specific project requirements or dataset characteristics.
"""

import os
import librosa
import numpy as np
import soundfile as sf
import multiprocessing as mp
import argparse


def load_audio(file_path, sample_rate=22050):
    """ Load an audio file using librosa. """
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr

def add_noise(audio, snr):
    """ Add random Gaussian noise to an audio file at a specified SNR. """
    rms_signal = np.sqrt(np.mean(audio**2))
    rms_noise = np.sqrt(rms_signal**2 / (10**(snr / 10)))
    noise = np.random.normal(0, rms_noise, audio.shape)
    return audio + noise

def change_pitch(audio, sample_rate, steps):
    """ Pitch shifting of the audio. """
    return librosa.effects.pitch_shift(audio, sample_rate, steps)

def time_stretch(audio, rate):
    """ Stretch the time of the audio. """
    return librosa.effects.time_stretch(audio, rate)

def process_file(file_path, output_dir, snr, pitch_step, stretch_rate, sample_rate=22050):
    """ Process a single file with noise addition, pitch shift, and time stretch. """
    audio, sr = load_audio(file_path, sample_rate)
    noisy_audio = add_noise(audio, snr)
    pitched_audio = change_pitch(audio, sr, pitch_step)
    stretched_audio = time_stretch(audio, stretch_rate)

    base_name = os.path.basename(file_path)
    sf.write(os.path.join(output_dir, f'noisy_{base_name}'), noisy_audio, sr)
    sf.write(os.path.join(output_dir, f'pitched_{base_name}'), pitched_audio, sr)
    sf.write(os.path.join(output_dir, f'stretched_{base_name}'), stretched_audio, sr)

def main(input_dir, output_dir, snr, pitch_step, stretch_rate):
    """ Main function to load files and process them with specified augmentations. """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.wav')]
    pool = mp.Pool(processes=mp.cpu_count())
    tasks = [(file, output_dir, snr, pitch_step, stretch_rate) for file in files]
    pool.starmap(process_file, tasks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Audio Data Augmentation")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save augmented audio files')
    parser.add_argument('--snr', type=float, default=10, help='Signal-to-Noise Ratio for noise addition')
    parser.add_argument('--pitch_step', type=int, default=0, help='Steps for pitch shifting')
    parser.add_argument('--stretch_rate', type=float, default=1.0, help='Rate for time stretching')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.snr, args.pitch_step, args.stretch_rate)
