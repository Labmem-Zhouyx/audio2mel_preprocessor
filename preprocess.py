import argparse
import os
import json
from multiprocessing import cpu_count

from tqdm import tqdm
from datasets import ljspeech, databaker, multisets


def write_metadata(metadata, out_dir, sr):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')
	mel_frames = sum([int(m[4]) for m in metadata])
	timesteps = sum([int(m[3]) for m in metadata])
	hours = timesteps / sr / 3600
	print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
		len(metadata), mel_frames, timesteps, hours))
	print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
	print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
	print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))


def main():
	print('Initializing preprocessing..')
	parser = argparse.ArgumentParser()
	parser.add_argument('--indir', required=True)
	parser.add_argument('--dataset', default='MultiSets')
	parser.add_argument('--outdir', default='training_data')
	parser.add_argument('--config_file', default='./datasets/config16k.json')
	parser.add_argument('--n_jobs', type=int, default=cpu_count())
	args = parser.parse_args()
	
	# Prepare directories
	in_dir = args.indir
	out_dir = args.outdir
	mel_dir = os.path.join(out_dir, 'mels')
	wav_dir = os.path.join(out_dir, 'audio')
	lin_dir = os.path.join(out_dir, 'linear')
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(lin_dir, exist_ok=True)
	# Process dataset
	if args.dataset == 'LJSpeech':
		metadata = ljspeech.build_from_path(in_dir, mel_dir, lin_dir, wav_dir, args.config_file, args.n_jobs, tqdm=tqdm)
	elif args.dataset == 'DataBaker':
		use_prosody = True
		metadata = databaker.build_from_path(in_dir, use_prosody, mel_dir, lin_dir, wav_dir, args.config_file, args.n_jobs, tqdm=tqdm)
	elif args.dataset == 'MultiSets':
		metadata = multisets.build_from_path(in_dir, mel_dir, lin_dir, wav_dir, args.config_file, args.n_jobs, tqdm=tqdm)
	else:
		raise ValueError('Unsupported dataset provided: {} '.format(args.dataset))
	
	# Write metadata to 'train.txt' for training
	with open(args.config_file, 'r') as f:
		sr = json.load(f)["sr"]
	write_metadata(metadata, out_dir, sr)


if __name__ == '__main__':
	main()
