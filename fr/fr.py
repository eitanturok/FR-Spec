from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
from tqdm import tqdm
import torch
import argparse
import os

from icecream import ic

def main(args):
	# load dataset and tokenizer
	ds = load_dataset(args.data_path, split=args.data_split, streaming=True)
	tokenizer = AutoTokenizer.from_pretrained(args.model_path)

	# count how often each token occurs in the dataset
	tok_freq = torch.zeros(tokenizer.vocab_size, dtype=torch.float32)
	for i, d in tqdm(enumerate(ds)):
		tokens = tokenizer.encode(d['text'], return_tensors='pt').squeeze()
		valid_tokens = tokens[tokens < tokenizer.vocab_size]
		if len(valid_tokens) > 0:
			tok_freq += torch.bincount(valid_tokens, minlength=tokenizer.vocab_size)
		if i == args.num_samples:
			break

	num_tokens = tok_freq.sum().astype(int)
	print(f"processed {args.num_samples} data samples and {num_tokens} tokens")

	# compute relative token frequency
	tok_rel_freq = tok_freq / num_tokens

	# save tok_rel_freq
	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	model_name = args.model_path.replace('/', '-')
	out_path = f'{args.out_dir}/{model_name}_tok_rel_freq.pt'
	with open(out_path, 'wb') as f:
		torch.save(tok_rel_freq, f)

	return tok_freq

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
