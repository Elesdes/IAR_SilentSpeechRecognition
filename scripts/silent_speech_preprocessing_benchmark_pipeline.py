"""
This script provides comprehensive functionality for preprocessing and feature extraction
from Electromyographic (EMG), audio, and textual data for silent speech recognition.

It includes:
- Data loading and verification of dataset structure.
- Preprocessing for EMG signals (e.g., bandpass filtering, notch filtering, artifact removal).
- Preprocessing for audio signals (e.g., noise reduction, pre-emphasis filtering).
- Text preprocessing (e.g., tokenization, feature extraction using TF-IDF and BERT embeddings).
- Feature extraction from EMG, audio, and textual data.
- Storage of extracted features in suitable formats for downstream tasks.

The code is structured into multiple classes:
- `Paths`: Manages and validates paths to dataset and preprocessed data directories.
- `EMGPreprocessor`: Handles preprocessing and feature extraction for EMG data.
- `AudioPreprocessor`: Handles preprocessing and feature extraction for audio data.
- `TextPreprocessor`: Handles textual preprocessing and feature extraction.

Finally, the script includes functionality to process a dataset directory, extract
features, and store them in structured formats. The entire dataset is archived
once processing is complete.

Credits:
This script is adapted and extended from the repository:
https://github.com/dgaddy/silent_speech

Author: Juan MAUBRAS
"""

import logging
from dataclasses import dataclass
import os
from typing import Tuple, Dict, Any, Literal, List
import librosa
import json
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch
import pywt
from sklearn.decomposition import FastICA
import noisereduce as nr
import re
import nltk
import shutil
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
from datetime import datetime

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

nltk.download("punkt_tab", quiet=True)


@dataclass
class Paths:
    data_dir: str = "data"
    dataset_dir: str = f"{data_dir}/silent_speech_dataset"
    preprocess_dir: str = f"{data_dir}/preprocessed_data"

    def __post_init__(self):
        os.makedirs(self.preprocess_dir, exist_ok=True)
        if not self.check_path_exists():
            raise FileNotFoundError("One or more paths do not exist.")
        logging.info("All paths exist.")

    def check_path_exists(self) -> bool:
        paths_exist = True
        for path in self.__dict__.values():
            if not os.path.exists(path):
                logging.warning(f"Path {path} does not exist.")
                paths_exist = False
        return paths_exist


paths = Paths()


# Preprocessing Configurations
@dataclass
class EMGPreprocessingConfig:
    apply_emg_bandpass: bool = True
    apply_emg_notch: bool = True
    apply_emg_rectification: bool = True
    apply_emg_smoothing: bool = True
    apply_emg_artifact_removal: bool = True
    emg_artifact_removal_method: Literal["wavelet", "ica", None] = "wavelet"
    emg_smoothing_window_size: int = 5  # ms
    emg_bandpass_lowcut: float = 20.0  # Hz
    emg_bandpass_highcut: float = 500.0  # Hz
    emg_notch_freq: float = 50.0  # Hz
    emg_fs: int = 1000  # Hz (sampling rate of EMG data)


@dataclass
class AudioPreprocessingConfig:
    use_clean_audio: bool = False  # If True, use clean audio; if False, use noisy audio
    apply_audio_preemphasis: bool = True
    apply_audio_noise_reduction: bool = True
    audio_noise_reduction_method: Literal["spectral_gate", None] = "spectral_gate"
    audio_preemphasis_coeff: float = 0.97
    audio_fs: int | None = None


@dataclass
class PreprocessingConfig(EMGPreprocessingConfig, AudioPreprocessingConfig):
    pass


config = PreprocessingConfig()


# EMG Preprocessing Class
class EMGPreprocessor:
    def __init__(self, config: EMGPreprocessingConfig):
        self.config = config

    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        nyquist = 0.5 * self.config.emg_fs
        highcut = min(self.config.emg_bandpass_highcut, nyquist - 1e-5)
        low = self.config.emg_bandpass_lowcut / nyquist
        high = highcut / nyquist
        if not (0 < low < high < 1):
            raise ValueError(
                f"Invalid cutoff frequencies: low={low}, high={high}. Must satisfy 0 < low < high < 1."
            )
        b, a = butter(4, [low, high], btype="band")
        y = filtfilt(b, a, data, axis=0)
        return y

    def notch_filter(self, data: np.ndarray) -> np.ndarray:
        nyquist = 0.5 * self.config.emg_fs
        w0 = self.config.emg_notch_freq / nyquist
        b, a = iirnotch(w0, 30.0)
        y = filtfilt(b, a, data, axis=0)
        return y

    def rectify_signal(self, data: np.ndarray) -> np.ndarray:
        return np.abs(data)

    def moving_average(self, data: np.ndarray) -> np.ndarray:
        window = (
            np.ones(self.config.emg_smoothing_window_size)
            / self.config.emg_smoothing_window_size
        )
        smoothed_data = np.apply_along_axis(
            lambda m: np.convolve(m, window, mode="same"), axis=0, arr=data
        )
        return smoothed_data

    def remove_artifacts_ica(
        self, data: np.ndarray, n_components: int = None
    ) -> np.ndarray:
        if n_components is None:
            n_components = data.shape[1]
        ica = FastICA(n_components=n_components)
        transformed = ica.fit_transform(data)
        reconstructed = ica.inverse_transform(transformed)
        return reconstructed

    def wavelet_denoise(
        self, data: np.ndarray, wavelet: str = "db4", level: int = 1
    ) -> np.ndarray:
        denoised_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            coeffs = pywt.wavedec(data[:, i], wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-level])) / 0.6745
            uthresh = sigma * np.sqrt(2 * np.log(len(data)))
            coeffs[1:] = [
                pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs[1:]
            ]
            reconstructed = pywt.waverec(coeffs, wavelet)

            # Ensure the reconstructed data has the same length as the input data
            if len(reconstructed) > data.shape[0]:
                reconstructed = reconstructed[: data.shape[0]]
            elif len(reconstructed) < data.shape[0]:
                reconstructed = np.pad(
                    reconstructed,
                    (0, data.shape[0] - len(reconstructed)),
                    mode="constant",
                )

            denoised_data[:, i] = reconstructed
        return denoised_data

    def preprocess(self, emg_data: np.ndarray) -> np.ndarray:
        try:
            if self.config.apply_emg_bandpass:
                emg_data = self.bandpass_filter(emg_data)
            if self.config.apply_emg_notch:
                emg_data = self.notch_filter(emg_data)
            if self.config.apply_emg_rectification:
                emg_data = self.rectify_signal(emg_data)
            if self.config.apply_emg_smoothing:
                emg_data = self.moving_average(emg_data)
            if self.config.apply_emg_artifact_removal:
                if self.config.emg_artifact_removal_method == "wavelet":
                    emg_data = self.wavelet_denoise(emg_data)
                elif self.config.emg_artifact_removal_method == "ica":
                    emg_data = self.remove_artifacts_ica(emg_data)
            logging.info(f"EMG data processed successfully.")
        except Exception as e:
            logging.error(f"Error processing EMG data: {e}")
        return emg_data

    # EMG Feature Extraction Methods
    def mean_absolute_value(self, data: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(data), axis=0)

    def zero_crossings(self, data: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        crossings = np.diff(np.sign(data - threshold), axis=0)
        zc = np.sum(crossings != 0, axis=0)
        return zc

    def waveform_length(self, data: np.ndarray) -> np.ndarray:
        wl = np.sum(np.abs(np.diff(data, axis=0)), axis=0)
        return wl

    def slope_sign_changes(
        self, data: np.ndarray, threshold: float = 0.0
    ) -> np.ndarray:
        diff = np.diff(data, axis=0)
        ssc = np.sum(
            ((diff[:-1] * diff[1:]) < threshold)
            & (np.abs(diff[:-1] - diff[1:]) >= threshold),
            axis=0,
        )
        return ssc

    def willison_amplitude(
        self, data: np.ndarray, threshold: float = 0.01
    ) -> np.ndarray:
        diff = np.abs(np.diff(data, axis=0))
        wamp = np.sum(diff >= threshold, axis=0)
        return wamp

    def simple_square_integral(self, data: np.ndarray) -> np.ndarray:
        ssi = np.sum(data**2, axis=0)
        return ssi

    def mean_frequency(self, data: np.ndarray, fs: int) -> np.ndarray:
        f, Pxx = welch(data, fs=fs, axis=0)
        mnf = np.sum(f[:, None] * Pxx, axis=0) / np.sum(Pxx, axis=0)
        return mnf

    def median_frequency(self, data: np.ndarray, fs: int) -> np.ndarray:
        f, Pxx = welch(data, fs=fs, axis=0)
        cumulative_power = np.cumsum(Pxx, axis=0)
        total_power = cumulative_power[-1, :]
        med_freq = np.zeros(Pxx.shape[1])
        for i in range(Pxx.shape[1]):
            idx = np.where(cumulative_power[:, i] >= total_power[i] / 2)[0][0]
            med_freq[i] = f[idx]
        return med_freq

    def power_spectrum_density(
        self, data: np.ndarray, fs: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        f, Pxx = welch(data, fs=fs, axis=0)
        return f, Pxx

    def sample_entropy(
        self, data: np.ndarray, m: int = 2, r: float = 0.2
    ) -> np.ndarray:
        N = data.shape[0]
        results = []
        for i in range(data.shape[1]):
            x = data[:, i]

            def _phi(m):
                x_m = np.array([x[j : j + m] for j in range(N - m + 1)])
                C = (
                    np.sum(
                        np.max(np.abs(x_m[:, None] - x_m[None, :]), axis=2) <= r, axis=0
                    )
                    - 1
                )
                return np.sum(C) / (N - m + 1)

            sampen = -np.log(_phi(m + 1) / _phi(m))
            results.append(sampen)
        return np.array(results)

    def approximate_entropy(
        self, data: np.ndarray, m: int = 2, r: float = 0.2
    ) -> np.ndarray:
        N = data.shape[0]
        results = []
        for i in range(data.shape[1]):
            x = data[:, i]

            def _phi(m):
                x_m = np.array([x[j : j + m] for j in range(N - m + 1)])
                C = np.sum(
                    np.max(np.abs(x_m[:, None] - x_m[None, :]), axis=2) <= r, axis=0
                ) / (N - m + 1)
                return np.sum(np.log(C)) / (N - m + 1)

            approxen = _phi(m) - _phi(m + 1)
            results.append(approxen)
        return np.array(results)

    def higuchi_fd(self, data: np.ndarray, kmax: int = 10) -> np.ndarray:
        N = data.shape[0]
        results = []
        for i in range(data.shape[1]):
            x = data[:, i]
            L = []
            for k in range(1, kmax):
                Lk = []
                for m in range(k):
                    idxs = np.arange(m, N, k)
                    Lm = (
                        np.sum(np.abs(np.diff(x[idxs]))) * (N - 1) / (((N - m) / k) * k)
                    )
                    Lk.append(Lm)
                L.append(np.mean(Lk))
            lnL = np.log(L)
            lnk = np.log(1.0 / np.arange(1, kmax))
            fd = np.polyfit(lnk, lnL, 1)[0]
            results.append(fd)
        return np.array(results)


# Audio Preprocessing Class
class AudioPreprocessor:
    def __init__(self, config: AudioPreprocessingConfig):
        self.config = config

    def pre_emphasis_filter(self, audio_data: np.ndarray) -> np.ndarray:
        emphasized_audio = np.append(
            audio_data[0],
            audio_data[1:] - self.config.audio_preemphasis_coeff * audio_data[:-1],
        )
        return emphasized_audio

    def reduce_noise_spectral_gate(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        reduced_noise = nr.reduce_noise(
            y=audio_data, sr=sr, n_fft=2048, prop_decrease=1.0
        )
        return reduced_noise

    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        return audio_data / np.max(np.abs(audio_data))

    def preprocess(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        if self.config.apply_audio_preemphasis:
            audio_data = self.pre_emphasis_filter(audio_data)
        if (
            self.config.apply_audio_noise_reduction
            and self.config.audio_noise_reduction_method == "spectral_gate"
        ):
            audio_data = self.reduce_noise_spectral_gate(audio_data, sr)
        # Remove non-finite values (e.g., NaN, inf) before further processing
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
        return audio_data

    # Audio Feature Extraction Method
    def extract_audio_features(
        self, audio_data: np.ndarray, sr: int, n_mfcc: int = 13
    ) -> Dict[str, np.ndarray]:
        features = {}
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        features["MFCCs_Mean"] = np.mean(mfccs, axis=1)
        features["MFCCs_Std"] = np.std(mfccs, axis=1)
        # Spectrogram
        S = librosa.stft(audio_data)
        S_db = librosa.amplitude_to_db(np.abs(S))
        features["Spectrogram_Mean"] = np.mean(S_db, axis=1)
        features["Spectrogram_Std"] = np.std(S_db, axis=1)
        return features


class TextPreprocessor:
    @staticmethod
    def preprocess_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        tokens = nltk.tokenize.word_tokenize(text)
        return tokens

    @staticmethod
    def extract_json_features(info_data: Dict[str, Any]) -> Dict[str, Any]:
        features = {}
        # Extract and process chunk information
        chunks = info_data.get("chunks", [])
        if chunks:
            chunks_array = np.array(chunks)
            features["chunks_mean"] = np.mean(chunks_array, axis=0).tolist()
            features["chunks_std"] = np.std(chunks_array, axis=0).tolist()
            features["chunks_max"] = np.max(chunks_array, axis=0).tolist()
            features["chunks_min"] = np.min(chunks_array, axis=0).tolist()
        else:
            features["chunks_mean"] = [0.0, 0.0, 0.0]
            features["chunks_std"] = [0.0, 0.0, 0.0]
            features["chunks_max"] = [0.0, 0.0, 0.0]
            features["chunks_min"] = [0.0, 0.0, 0.0]

        # Extract and process text information
        text = info_data.get("text", "")
        if text.strip():
            processed_text = TextPreprocessor.preprocess_text(text)
            tokens = TextPreprocessor.tokenize_text(processed_text)
            features["text_token_count"] = len(tokens)
            features["text_processed"] = processed_text

            if len(tokens) > 0:
                try:
                    tfidf_vectorizer = TfidfVectorizer(max_features=100)
                    tfidf_features = tfidf_vectorizer.fit_transform([processed_text])
                    features["tfidf_features"] = tfidf_features.toarray().tolist()
                except ValueError as e:
                    # Handle empty vocabulary or stop words only error
                    features["tfidf_features"] = []

                # BERT Embedding Feature Extraction (Optional, for more sophisticated features)
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                model = BertModel.from_pretrained("bert-base-uncased")
                inputs = tokenizer(processed_text, return_tensors="pt")
                outputs = model(**inputs)
                # Taking the mean of the last hidden state to form a single feature vector
                text_embedding = (
                    torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()
                )
                features["bert_embedding"] = text_embedding.tolist()
        else:
            # Add an indication that text data was empty or missing
            features["text_token_count"] = 0
            features["text_processed"] = "N/A"
            features["tfidf_features"] = []
            features["bert_embedding"] = []

        return features


# Load Functions
def load_emg_data(sample_id: str, directory: str) -> np.ndarray | None:
    emg_path = os.path.join(directory, f"{sample_id}_emg.npy")
    if os.path.exists(emg_path):
        logging.info(f"Loading EMG data for sample {sample_id}.")
        return np.load(emg_path)
    else:
        logging.warning(f"EMG data for sample {sample_id} does not exist.")
        return None


def load_audio_data(
    sample_id: str, directory: str, clean: bool = False
) -> Tuple[np.ndarray | None, int | None]:
    suffix = "_audio_clean.flac" if clean else "_audio.flac"
    audio_path = os.path.join(directory, f"{sample_id}{suffix}")
    if os.path.exists(audio_path):
        logging.info(
            f"Loading Audio data for sample {sample_id} ({'clean' if clean else 'noisy'})."
        )
        return librosa.load(audio_path, sr=None)
    else:
        logging.warning(f"Audio data for sample {sample_id} does not exist.")
        return None, None


def load_info_data(sample_id: str, directory: str) -> Dict[str, Any] | None:
    info_path = os.path.join(directory, f"{sample_id}_info.json")
    if os.path.exists(info_path):
        logging.info(f"Loading Info data for sample {sample_id}.")
        with open(info_path, "r") as f:
            return json.load(f)
    else:
        logging.warning(f"Info data for sample {sample_id} does not exist.")
        return None


def get_preprocess_subdir(dataset_subdir: str) -> str:
    """
    Create a corresponding subdirectory in preprocess_dir for the given dataset subdirectory.
    """
    relative_subdir = os.path.relpath(dataset_subdir, paths.dataset_dir)
    preprocess_subdir = os.path.join(paths.preprocess_dir, relative_subdir)
    os.makedirs(preprocess_subdir, exist_ok=True)
    return preprocess_subdir


# Preprocessing and Saving Features
output_dir = paths.preprocess_dir


def process_and_store_data(
    sample_id: str,
    emg_dir: str,
    audio_dir: str,
    info_dir: str,
    emg_preprocessor: EMGPreprocessor,
    audio_preprocessor: AudioPreprocessor,
    config: PreprocessingConfig,
    output_subdir: str,
):
    try:
        # Load and preprocess EMG data
        emg_data = load_emg_data(sample_id, emg_dir)
        if emg_data is not None:
            emg_data = emg_preprocessor.preprocess(emg_data)
            emg_features = {
                "MAV": emg_preprocessor.mean_absolute_value(emg_data).tolist(),
                "ZC": emg_preprocessor.zero_crossings(emg_data).tolist(),
                "WL": emg_preprocessor.waveform_length(emg_data).tolist(),
                "SSC": emg_preprocessor.slope_sign_changes(emg_data).tolist(),
                "WAMP": emg_preprocessor.willison_amplitude(emg_data).tolist(),
                "SSI": emg_preprocessor.simple_square_integral(emg_data).tolist(),
            }
            emg_output_path = os.path.join(
                output_subdir, f"{sample_id}_emg_features.npz"
            )
            np.savez(emg_output_path, **emg_features)
            logging.info(
                f"EMG features for sample {sample_id} saved to {emg_output_path}."
            )

        # Load and preprocess audio data (both clean and noisy)
        for clean_flag in [True, False]:
            audio_data, sr = load_audio_data(sample_id, audio_dir, clean=clean_flag)
            if audio_data is not None:
                audio_data = audio_preprocessor.preprocess(audio_data, sr)
                suffix = "clean" if clean_flag else "noisy"
                audio_features = audio_preprocessor.extract_audio_features(
                    audio_data, sr
                )
                audio_features = {k: v.tolist() for k, v in audio_features.items()}
                audio_output_path = os.path.join(
                    output_subdir, f"{sample_id}_audio_features_{suffix}.npz"
                )
                np.savez(audio_output_path, **audio_features)
                logging.info(
                    f"Audio features ({suffix}) for sample {sample_id} saved to {audio_output_path}."
                )

        # Load and process info data (text and chunks)
        info_data = load_info_data(sample_id, info_dir)
        if info_data is not None:
            json_features = TextPreprocessor.extract_json_features(info_data)
            info_output_path = os.path.join(
                output_subdir, f"{sample_id}_info_features.json"
            )
            with open(info_output_path, "w") as out_f:
                json.dump(json_features, out_f)
            logging.info(
                f"Info features for sample {sample_id} saved to {info_output_path}."
            )

    except Exception as e:
        logging.error(f"Error processing data for sample {sample_id}: {e}")


def walk_and_process_dataset(dataset_dir: str, config: PreprocessingConfig):
    emg_preprocessor = EMGPreprocessor(config)
    audio_preprocessor = AudioPreprocessor(config)

    for root, _, files in tqdm(os.walk(dataset_dir), desc="Processing Dataset"):
        if not any(
            file.endswith(("_emg.npy", "_audio.flac", "_info.json")) for file in files
        ):
            continue

        output_subdir = get_preprocess_subdir(root)
        for file in files:
            if file.endswith("_emg.npy"):
                sample_id = file.split("_emg")[0]
                logging.info(f"Processing sample {sample_id} in directory {root}.")
                process_and_store_data(
                    sample_id,
                    root,
                    root,
                    root,
                    emg_preprocessor,
                    audio_preprocessor,
                    config,
                    output_subdir,
                )


walk_and_process_dataset(paths.dataset_dir, config)

shutil.make_archive(
    os.path.join(paths.data_dir, "preprocessed_dataset"), "zip", paths.preprocess_dir
)
logging.info("Preprocessing complete and data archived.")
