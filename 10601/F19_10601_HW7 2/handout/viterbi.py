from learnhmm import *
import sys



def new_dict(infile_path):
	res = dict()
	index = 0
	with open(infile_path, 'r') as f:
		for item in f:
			char = item.strip()
			res[index] = char
			index += 1
	return res


def viterbi(prior, trans, emit, example, len_tag_dict):
	predict = []
	W = np.zeros((len(example), len_tag_dict))
	B = np.zeros((len(example), len_tag_dict))
	for t in range(len(example)):
		if t == 0:
			W[t] = prior + emit[:, example[t]]
			for i in range(len_tag_dict):
				B[t][i] = i
		else:
			for k in range(len_tag_dict):
				W_t_minus_1 = W[t-1]
				tmp = []
				for j in range(len_tag_dict):
					tmp_val = W_t_minus_1[j] + trans[j][k] + emit[k][example[t]]
					tmp.append(tmp_val)
				index = np.argmax(tmp)
				B[t][k] = index
				W[t][k] = tmp[index]
	y_hat_T = np.argmax(W[len(example) - 1])
	predict.append(y_hat_T)
	for t in range(len(example) - 1, 0, -1):
		y_hat_T_minus_1 = B[t][int(y_hat_T)]
		predict.insert(0, y_hat_T_minus_1)
		y_hat_T = y_hat_T_minus_1
	res = []
	for i in range(len(predict)):
		res.append(int(predict[i]))
	return res




def main():
	test_input = sys.argv[1]
	index_to_word = sys.argv[2]
	index_to_tag = sys.argv[3]
	hmmprior = sys.argv[4]
	hmmemit = sys.argv[5]
	hmmtrans = sys.argv[6]
	predicted_file = sys.argv[7]
	metrics_file = sys.argv[8]

	words, tags = parse_data(test_input)
	tag_dict = load_index_dict(index_to_tag)
	word_dict = load_index_dict(index_to_word)
	tag_dict_new = new_dict(index_to_tag)
	word_dict_new = new_dict(index_to_word)
	words_idx_lst = convert_to_index(words, word_dict)
	tags_idx_lst = convert_to_index(tags, tag_dict)
	prior = np.loadtxt(hmmprior)
	trans = np.loadtxt(hmmtrans)
	emit = np.loadtxt(hmmemit)

	prior = np.log(prior)
	trans = np.log(trans)
	emit = np.log(emit)
	out_str = ""
	error = 0
	for example, tag_example in zip(words_idx_lst, tags_idx_lst):
		predict = viterbi(prior, trans, emit, example, len(tag_dict))
		for tag_ex, pred, in zip(tag_example, predict):
			if tag_ex != pred:
				error += 1
		sub_str = ""
		for word, tag in zip(example, predict):
			word_str = word_dict_new[word]
			pred_str = tag_dict_new[tag]
			sub_str += (word_str + "_" + pred_str + " ")
		sub_str = sub_str[:-1]
		out_str += sub_str + "\n"
	out_str = out_str[:-1]

	len_of_tag = 0
	for i in range(len(tags_idx_lst)):
		len_of_tag += len(tags_idx_lst[i])
	accuracy = (len_of_tag - error) / len_of_tag
	write(predicted_file, out_str)
	matrix_str = "Accuracy: " + str(accuracy)
	write(metrics_file, matrix_str)



if __name__ == '__main__':
	main()