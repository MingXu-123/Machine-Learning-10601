import numpy as np
import sys


def parse_data(train_in):
	words = []
	tags = []
	with open(train_in, 'r') as f:
		for line in f:
			words_lst = []
			for item in line.split():
				words_lst.append(item)
			tmp_words = []
			tmp_tags = []
			for item in words_lst:
				tmp_words.append(item.split('_')[0])
				tmp_tags.append(item.split('_')[1])
			words.append(tmp_words)
			tags.append(tmp_tags)
	return words, tags


def load_index_dict(infile_path):
	res = dict()
	index = 0
	with open(infile_path, 'r') as f:
		for item in f:
			char = item.strip()
			res[char] = index
			index += 1
	return res


def convert_to_index(item_lst, item_index_dict):
	res = []
	for i in range(len(item_lst)):
		char_lst = item_lst[i]
		tmp = []
		for i in range(len(char_lst)):
			char = char_lst[i]
			char_lst[i] = item_index_dict[char]
			tmp.append(char_lst[i])
		res.append(tmp)
	return res


def calculate_hmmemit(tags_idx_lst, words_idx_lst, word_dict, tag_dict):
	emit = np.ones((len(tag_dict), len(word_dict)), dtype = float)
	for i in range(len(tags_idx_lst)):
		tag_example = tags_idx_lst[i]
		word_example = words_idx_lst[i]
		for tag, word in zip(tag_example, word_example):
			j = tag
			k = word
			emit[j][k] += 1
	for i in range(len(emit)):
		total = sum(emit[i])
		emit[i] /= total
	return emit
 


def calculate_hmmprior(tags_idx_lst, tag_dict):
	counter_dict = {}
	for tags in tags_idx_lst:
		if tags[0] not in counter_dict:
			counter_dict[tags[0]] = 1
		else:
			counter_dict[tags[0]] += 1
	prior_vec = np.ones((len(tag_dict), 1), dtype = float)
	total = len(tags_idx_lst) + len(tag_dict)
	for key, value in counter_dict.items():
		prior_vec[key] += value
	prior_vec /= total
	return prior_vec


def calculate_hmmtrans(tags_idx_lst, tag_dict):
	col, row = len(tag_dict), len(tag_dict)
	trans_matrix = np.ones((row, col), dtype = float)
	for tag_lst in tags_idx_lst:
		for i in range(len(tag_lst) - 1):
			j = tag_lst[i]
			k = tag_lst[i + 1]
			trans_matrix[j][k] += 1
	for i in range(len(trans_matrix)):
		total = sum(trans_matrix[i])
		trans_matrix[i] /= total
	return trans_matrix


def generate_str(matrix):
	res = ""
	for row in matrix:
		for num in row:
			num = "%.18e"%(num)
			res += (str(num) + " ")
		res = res[:-1]
		res += "\n"
	res = res[:-1]
	return res


def write(out_path, str_output):
    with open(out_path, 'w') as f:
        f.write(str_output)



def main():
	train_in = sys.argv[1]
	index_to_word = sys.argv[2]
	index_to_tag = sys.argv[3]
	hmmprior = sys.argv[4]
	hmmemit = sys.argv[5]
	hmmtrans = sys.argv[6]
	words, tags = parse_data(train_in)
	tag_dict = load_index_dict(index_to_tag)
	word_dict = load_index_dict(index_to_word)
	words_idx_lst = convert_to_index(words, word_dict)
	tags_idx_lst = convert_to_index(tags, tag_dict)
	prior = calculate_hmmprior(tags_idx_lst, tag_dict)
	trans = calculate_hmmtrans(tags_idx_lst, tag_dict)
	emit = calculate_hmmemit(tags_idx_lst, words_idx_lst, word_dict, tag_dict)
	prior_str = generate_str(prior)
	emit_str = generate_str(emit)
	trans_str = generate_str(trans)
	write(hmmprior, prior_str)
	write(hmmemit, emit_str)
	write(hmmtrans, trans_str)


if __name__ == '__main__':
	main()
