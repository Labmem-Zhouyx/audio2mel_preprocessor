from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import os
import re
from datasets import mel


def build_from_path(input_dir, use_prosody, mel_dir, linear_dir, wav_dir, config_file, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the AISHELL-3 dataset from a gven input path to given output directories
	(https://www.openslr.org/resources/93/data_aishell3.tgz)

		AISHELL-3/train(or test)
			├── wav              (dir of wav files)
			|    └──SSBxxxx      (dir of speaker id, storing *.wav files)
			├── content.txt      (storing original text and corresponding wav path)
			└── *label_train-set.txt (storing text with prosody label and corresponding wav path, only for train)

	Args:
		- input_dir: input directory that contains the files to prerocess
		- use_prosody: whether use the information of "label_train-set.txt", only for "train"
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- config_file: the config file for mel extractor
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. This should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	content = _read_files(os.path.join(input_dir, 'content.txt'))
	for idx in range(len(content)):
		res = _parse_cn_text(content[idx])
		if res is not None:
			basename, text, text_pinyin = res
			speakerid = basename[:7]
			wav_path = os.path.join(input_dir, 'wav', speakerid, '{}.wav'.format(basename))
			futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, text_pinyin, speakerid, config_file)))

	return [future.result() for future in tqdm(futures) if future.result() is not None]


def _read_files(file):
	labels = []
	with open(file, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line.startswith('#') and line != '': labels.append(line)
	return labels


def _parse_cn_text(text):
	"""
	Parse label from text and pronunciation lines with prosodic structure labelings
	
	Input text:    SSB00050003.wav	七 qi1 路 lu4 无 wu2 人 ren2 售 shou4 票 piao4
	Return sen_id: SSB00050003
	Return text_aft: 七路无人售票
	Return pinyin: qi1 lu4 wu2 ren2 shou4 piao4
	"""

	text = text.strip()

	if len(text) == 0:
		return None

	# split into sub-terms
	sen_id, texts = text.split('\t')

	sen_id = sen_id.split('.')[0]
	texts = texts.split(' ')

	pinyin = ''
	text_aft = ''
	i = 0
	while i < len(texts):
		text_aft += texts[i]
		i += 1
		pinyin += texts[i] + ' '
		i += 1 

	return (sen_id, text_aft, pinyin)


def _process_utterance(mel_dir, linear_dir, wav_dir, index, wav_path, text, text_pinyin, speaker_id, config_file):
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
	return (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text, text_pinyin, speaker_id)
