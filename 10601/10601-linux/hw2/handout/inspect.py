import sys
import csv
import math
import numpy as np


def get_label_info(label_dict):
	label1 = ""
	label2 = ""
	for key in label_dict:
		if label1 == "":
			label1 = key
		else:
			label2 = key
	num_of_label1 = label_dict[label1]
	num_of_label2 = 0
	if label2 == "":
		num_of_label2 = 0
	else:
		num_of_label2 = label_dict[label2]
	num_of_sum = num_of_label1 + num_of_label2
	return num_of_label1, num_of_label2, num_of_sum


def calculate_entropy(label_dict):
	if len(label_dict) == 0:
		return 0
	num_of_label1, num_of_label2, num_of_sum \
	 = get_label_info(label_dict)
	prob_of_label1 = num_of_label1 / num_of_sum
	prob_of_label2 = num_of_label2 / num_of_sum
	if (prob_of_label1 == 0):
		return 0
	if (prob_of_label2 == 0):
		return 0
	log_label1 = math.log(prob_of_label1, 2)
	log_label2 = math.log(prob_of_label2, 2)
	entropy = - (prob_of_label1 * log_label1) \
	          - (prob_of_label2 * log_label2)
	return entropy


def calculate_error_rate(label_dict):
	num_of_label1, num_of_label2, num_of_sum \
	 = get_label_info(label_dict)
	majority_of_vote = max(num_of_label1, num_of_label2)
	error_rate = 0
	if (majority_of_vote == num_of_label1):
		error_rate = num_of_label2 / num_of_sum
	else:
		error_rate = num_of_label1 / num_of_sum
	return error_rate


def get_label(input_file):
	count_Dict = dict()
	with open(input_file, 'r') as tsvfile:
		reader = csv.reader(tsvfile)
		count = 0
		for row in reader:
			if count == 0:
				count += 1
				continue
			parse_string = row[0]
			tmplst = []
			for sub_string in parse_string.split("\t"):
				tmplst.append(sub_string)
			label = tmplst[-1]
			if label not in count_Dict:
				count_Dict[label] = 1
			else:
				count_Dict[label] += 1
	return count_Dict


def writeFile(path, contents1, contents2):
    with open(path, "w") as f:
        f.write("entropy: " + contents1 + "\n")
        f.write("error: " + contents2)


def calculate_entropy_and_error_rate_main(output_file, label_dict):
	entropy_of_labels = calculate_entropy(label_dict)
	error_rate = calculate_error_rate(label_dict)
	writeFile(output_file, str(entropy_of_labels), str(error_rate))


def main():
	input_file = sys.argv[1]
	output_file = sys.argv[2]
	label_dict = get_label(input_file)
	calculate_entropy_and_error_rate_main(output_file, label_dict)


if __name__ == "__main__":
    main()



