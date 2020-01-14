import sys


def model_2_helper(review_text, words_dict):
	threshold = 4
	text_dict = dict()
	res_dict = dict()
	review_text = review_text.split(" ")
	for string in review_text:
		if string in words_dict:
			if string not in text_dict:
				text_dict[string] = 1
			else:
				text_dict[string] += 1
		elif string not in words_dict:
			continue
	for word in text_dict:
		if text_dict[word] < threshold:
			res_dict[word] = 1
	return res_dict


def model_1_helper(review_text, words_dict):
	text_dict = dict()
	review_text = review_text.split(" ")
	for string in review_text:
		if string in words_dict:
			if string not in text_dict:
				text_dict[string] = 1
			else:
				continue
		elif string not in words_dict:
			continue
	return text_dict


def model_1_and_2(input_path, words_dict, formatted_output, feature_flag):
	res = ""
	res_lst = []
	with open (input_path, 'r') as f:
		line_of_raw_text = f.readlines()
		for text in line_of_raw_text:
			text = text[:-1]
			label, review_text = text.split("\t")
			if feature_flag == 1:
				feature_vec = model_1_helper(review_text, words_dict)
				res_lst.append((label, feature_vec))
			elif feature_flag == 2:
				feature_vec = model_2_helper(review_text, words_dict)
				res_lst.append((label, feature_vec))
	for feature_pair in res_lst:
		label = feature_pair[0]
		feature_dict = feature_pair[1]
		res += str(label) + "\t"
		for feature in feature_dict:
			res += str(words_dict[feature]) + ":" + str(feature_dict[feature]) + "\t"
		res = res[:-1] + "\n"
	with open (formatted_output, 'w') as output:
		output.write(res)


def create_formatted_data(input_path, words_dict, formatted_output, feature_flag):
	model_1_and_2(input_path, words_dict, formatted_output, feature_flag)


def create_words_dict(dict_input):
	words_dict = dict()
	with open (dict_input, 'r') as f:
		line_of_raw_dict = f.readlines()
		for line in line_of_raw_dict:
			line = line[:-1]
			line_lst = line.split(" ")
			words_dict[line_lst[0]] = line_lst[1]
	return words_dict


def main():
	train_input = sys.argv[1]
	vaild_input = sys.argv[2]
	test_input = sys.argv[3]
	dict_input = sys.argv[4]
	train_formatted_output = sys.argv[5]
	valid_formatted_output = sys.argv[6]
	test_formatted_output = sys.argv[7]
	feature_flag = int(sys.argv[8])
	words_dict = create_words_dict(dict_input)
	# feature_flag can 1 or 2
	create_formatted_data(train_input, words_dict, train_formatted_output, feature_flag)
	create_formatted_data(vaild_input, words_dict, valid_formatted_output, feature_flag)
	create_formatted_data(test_input, words_dict, test_formatted_output, feature_flag)


if __name__ == "__main__":
	main()
