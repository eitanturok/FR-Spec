from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
from tqdm import tqdm
import torch
import argparse
import os


def main(args):
	# load dataset and tokenizer
	ds = load_dataset(args.data_path, split=args.data_split, streaming=True)
	tokenizer = AutoTokenizer.from_pretrained(args.model_path)

	# count how often each token occurs in the dataset
	vocab_freqs = torch.zeros(tokenizer.vocab_size, dtype=torch.float32)
	for i, d in tqdm(enumerate(ds), total=args.num_samples):
		tokens = tokenizer.encode(d['text'], return_tensors='pt').squeeze()
		valid_tokens = tokens[tokens < tokenizer.vocab_size]
		if len(valid_tokens) > 0:
			vocab_freqs += torch.bincount(valid_tokens, minlength=tokenizer.vocab_size)
		if i == args.num_samples:
			break

	num_tokens = int(vocab_freqs.sum())
	print(f"processed {args.num_samples} data samples and {num_tokens} tokens")

	# save token_scores
	os.makedirs(args.out_dir, exist_ok=True)
	model_name = args.model_path.replace('/', '-').lower()
	out_path = f'{args.out_dir}/{model_name}_vocab_freqs.pt'
	with open(out_path, 'wb') as f:
		torch.save(vocab_freqs, f)
	print(f'Saved token scores at {out_path}')

	return vocab_freqs

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# unsloth/Llama-3.2-1B-Instruct is a reupload of meta-llama/Meta-Llama-3.2-1B-Instruct that doesn't require HF token
	parser.add_argument(
		'--model_path',
		type=str,
		default='unsloth/Llama-3.2-1B-Instruct',
		help='The path to the model.'
	)
	parser.add_argument(
		'--data_path',
		type=str,
		default='cerebras/SlimPajama-627B',
		help='The path to the dataset.'
	)
	parser.add_argument(
		'--data_split',
		type=str,
		default='train',
		help='The split of the dataset.'
	)
	parser.add_argument(
		'--num_samples',
		type=int,
		default=10,
		help='The number of dataset samples to process.'
	)
	parser.add_argument(
		'--out_dir',
		type=str,
		default='./fr-index',
		help='The path to save the sorted tokens and frequencies.'
	)

	args = parser.parse_args()
	print(args)
	main(args)
