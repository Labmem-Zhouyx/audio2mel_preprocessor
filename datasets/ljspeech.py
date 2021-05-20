from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import os
from datasets import mel


def build_from_path(input_dir, mel_dir, linear_dir, wav_dir, config_file, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the LJ speech format dataset from a gven input path to given output directories

		LJSpeech-1.1
			├── metadata.csv ('|' separated metadata: wav_file_name|raw text|text)
			└── wavs         (dir storing wav files)

	Args:
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- config_file: the config file for mel extractor
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	index = 1
	with open(os.path.join(input_dir, 'metadata.csv'), encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split('|')
			basename = parts[0]
			wav_path = os.path.join(input_dir, 'wavs', '{}.wav'.format(basename))
			text = parts[2]
			futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, config_file)))
			index += 1

	return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(mel_dir, linear_dir, wav_dir, index, wav_path, text, config_file):
	"""
	Preprocesses a single utterance wav/text pair

	this writes the mel scale spectogram to disk and return a tuple to write
	to the train.txt file

	Args:
		- mel_dir: the directory to write the mel spectograms into
		- linear_dir: the directory to write the linear spectrograms into
		- wav_dir: the directory to write the preprocessed wav into
		- index: the numeric index to use in the spectogram filename
		- wav_path: path to the audio file containing the speech input
		- text: text spoken in the input audio file
		- config_file: the config file for mel extractor

	Returns:
		- A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
	"""
	if os.path.exists(wav_path):
		mel_spectrogram, linear_spectrogram, audio = mel.wav2mel_config(wav_path, config_file)
		time_steps = len(audio)
		mel_frames = mel_spectrogram.shape[0]
	else:
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(wav_path))
		return None

	# Write the spectrogram and audio to disk
	audio_filename = 'audio-{}.npy'.format(index)
	mel_filename = 'mel-{}.npy'.format(index)
	linear_filename = 'linear-{}.npy'.format(index)
	np.save(os.path.join(wav_dir, audio_filename), audio.astype(np.float32), allow_pickle=False)
	np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram, allow_pickle=False)
	np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram, allow_pickle=False)

	# Return a tuple describing this training example
	return (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text)
