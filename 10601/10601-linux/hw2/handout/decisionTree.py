import sys
import random
import copy
import csv
import math
import numpy as np
from inspect import calculate_entropy, calculate_error_rate, get_label_info


class DecisionTreeRootNode(object):
	def __init__(self, split_attribute = None, vote = None, depth = None, left = None, right = None):
		self.split_attribute = split_attribute
		self.vote = vote
		self.left = left
		self.right = right
		self.depth = depth
		self.left_children_value = None
		self.right_children_value = None
		self.label_dict = None


def parse_str(parse_string, lst):
	for sub_string in parse_string.split("\t"):
					lst.append(sub_string)


def Inport_data(inputfile):
	with open(inputfile, 'r') as inputdata:
		reader = csv.reader(inputdata, delimiter = ',')
		dataset = []
		attributes = []
		observations = []
		count = 0
		for row in reader:
			if count == 0:
				count += 1
				parse_string_attribute = row[0]
				parse_str(parse_string_attribute, attributes)
			else:
				parse_string = row[0]
				tmplst = []
				parse_str(parse_string, tmplst)
				observations.append(tmplst)
		dataset += [attributes]
		dataset += observations
		return dataset


def create_label_dict(observations):
	label_dict = {}
	for observation in observations:
		label = observation[-1]
		if label not in label_dict:
			label_dict[label] = 1
		else:
			label_dict[label] += 1
	return label_dict


def calculate_mutual_information(entropy_of_label, conditional_entropy_of_label):
	return entropy_of_label - conditional_entropy_of_label


def find_possible_value_of_a_feature(index_of_feature, observations):
	s = set() # find possible value of a feature
	for observation in observations:
		s.add(observation[index_of_feature])
	s = list(s)
	if (len(s) == 2):
		return s[0], s[1]
	else:
		return s[0], ""


def split_dataset(index_of_feature, observations, value_of_feature1, value_of_feature2):
	subset1 = []
	subset2 = []
	for observation in observations:
		if (observation[index_of_feature]) == value_of_feature1:
			subset1.append(observation)
		elif (observation[index_of_feature] == value_of_feature2):
			subset2.append(observation)
	return subset1, subset2


def calculate_conditional_entropy(feature, index_of_feature, observations):
	value_of_feature1, value_of_feature2 = \
	find_possible_value_of_a_feature(index_of_feature, observations)
	subset1, subset2 = split_dataset(index_of_feature, observations,
	                                 value_of_feature1, value_of_feature2)
	labels_dict_of_subset1 = create_label_dict(subset1)
	labels_dict_of_subset2 = create_label_dict(subset2)
	entropy_of_subset1 = calculate_entropy(labels_dict_of_subset1)
	entropy_of_subset2 = calculate_entropy(labels_dict_of_subset2) 
	prob_of_value_of_feature1 = len(subset1) / len(observations)
	prob_of_value_of_feature2 = len(subset2) / len(observations)
	conditional_entropy = (prob_of_value_of_feature1 * entropy_of_subset1) + \
	                      (prob_of_value_of_feature2 * entropy_of_subset2)
	return conditional_entropy


def select_the_best_feature(mutual_information_lst):
	if (len(mutual_information_lst) == 1):
		return mutual_information_lst[0][0]
	best_feature = ""
	mutual_information_max = 0
	for item in mutual_information_lst:
		if item[1] >= mutual_information_max:
			mutual_information_max = item[1]
			best_feature = item[0]
	if mutual_information_max == 0:
		return None
	return best_feature


def majority_vote_classifier(observations):
	label_dict = create_label_dict(observations)
	vote = None
	tmp_vote1 = None
	tmp_vote2 = None
	for key in label_dict:
		if tmp_vote1 == None:
			tmp_vote1 = key
		else:
			tmp_vote2 = key
	numof_vote1 = label_dict[tmp_vote1]
	numof_vote2 = label_dict[tmp_vote2]
	max_vote = max(numof_vote1, numof_vote2)
	if (max_vote == numof_vote1) and (max_vote == numof_vote2):
		vote = random.choice([tmp_vote1, tmp_vote2])
	elif max_vote == numof_vote1:
		vote = tmp_vote1
	elif max_vote == numof_vote2:
		vote = tmp_vote2
	return vote


def perfect_classified_vote(observations):
	label_dict = create_label_dict(observations)
	vote = None
	for key in label_dict:
		vote = key
	return vote


def data_is_unambiguous(observations):
	label_dict = create_label_dict(observations)
	if len(label_dict) == 1:
		return True


def no_remaining_features(remaining_features):
	return len(remaining_features) == 0


def build_tree(features_column, remaining_features, observations, 
	           max_depth, current_depth, value_of_label1, value_of_label2):
	label_dict = create_label_dict(observations)
	if value_of_label1 not in label_dict:
		label_dict[value_of_label1] = 0
	if value_of_label2 not in label_dict:
		label_dict[value_of_label2] = 0
	if len(observations) == 0:
			return None
	if data_is_unambiguous(observations):
		leaf_node = DecisionTreeRootNode()
		vote = perfect_classified_vote(observations)
		leaf_node.vote = vote
		leaf_node.depth = current_depth
		leaf_node.label_dict = label_dict
		return leaf_node
	if no_remaining_features(remaining_features) or (current_depth == max_depth):
		leaf_node = DecisionTreeRootNode()
		leaf_node.depth = current_depth
		leaf_node.vote = majority_vote_classifier(observations)
		leaf_node.label_dict = label_dict
		return leaf_node
	else:
		best_feature = None
		entropy_of_label = calculate_entropy(label_dict) #H(Y)
		mutual_information_lst = []
		for feature in remaining_features:
			index_of_feature = features_column.index(feature)
			conditional_entropy_of_label = \
			calculate_conditional_entropy(feature, index_of_feature, observations)  #H(Y|A)
			information_gain = \
			calculate_mutual_information(entropy_of_label, conditional_entropy_of_label)
			mutual_information_lst.append((feature, information_gain))
		best_feature = select_the_best_feature(mutual_information_lst)
		if best_feature == None: # case that MI max == 0, so return leaf node(majority)
			leaf_node = DecisionTreeRootNode()
			leaf_node.depth = current_depth
			leaf_node.vote = majority_vote_classifier(observations)
			leaf_node.label_dict = label_dict
			return leaf_node
		index_of_best_feature = features_column.index(best_feature)
		value_of_feature1, value_of_feature2 = \
		find_possible_value_of_a_feature(index_of_best_feature, observations)
		subset1, subset2 = split_dataset(index_of_best_feature, observations,
	                                 value_of_feature1, value_of_feature2)
		remaining_features_rec = copy.deepcopy(remaining_features)
		remaining_features_rec.remove(best_feature) 
		left_node = build_tree(features_column, remaining_features_rec, 
	                           subset1, max_depth, current_depth + 1, 
	                           value_of_label1, value_of_label2)
		right_node = build_tree(features_column, remaining_features_rec, 
	                           subset2, max_depth, current_depth + 1, 
	                           value_of_label1, value_of_label2)
		vote = None  
		node = DecisionTreeRootNode(best_feature, vote, current_depth, left_node, right_node)
		node.left_children_value = value_of_feature1
		node.right_children_value = value_of_feature2 
		node.label_dict = label_dict
		return node


def pretty_print2(node, value_of_label1, value_of_label2):
	label_dict = node.label_dict
	num_of_label1 = label_dict[value_of_label1]
	num_of_label2 = label_dict[value_of_label2]
	string1 = "[" + str(num_of_label1) + " " + str(value_of_label1) + \
		          " /" + str(num_of_label2) + " " + str(value_of_label2) + "]"
	print(string1)

def pretty_print(node, value_of_label1, value_of_label2):
	if node == None:
		return
	if node.vote != None:
		return
	else:
		if node.left == None:
			return
		if node.right == None:
			return
		label_dict = node.label_dict
		num_of_label1 = label_dict[value_of_label1]
		num_of_label2 = label_dict[value_of_label2]
		string1 = "[" + str(num_of_label1) + " " + str(value_of_label1) + \
		          " /" + str(num_of_label2) + " " + str(value_of_label2) + "]"
		if node.depth == 0:
			print(string1)
		string2 = ""
		if node.split_attribute != None:
			num_of_depth = node.depth + 1
			for i in range(num_of_depth):
				string2 += "| "
			string3 = string2
			string3 += (str(node.split_attribute) + " = " + str(node.left_children_value) + ": ")
			left_children_label_dict = node.left.label_dict
			num_of_child_label1 = left_children_label_dict[value_of_label1]
			num_of_child_label2 = left_children_label_dict[value_of_label2]
			string4 = "[" + str(num_of_child_label1) + " " + str(value_of_label1) + \
			          " /" + str(num_of_child_label2) + " " + str(value_of_label2) + "]"
			print(string3 + string4)

		pretty_print(node.left, value_of_label1, value_of_label2)
		new_str = string2 + node.split_attribute + " ="

		right_children_label_dict = node.right.label_dict
		num_of_child_label1 = right_children_label_dict[value_of_label1]
		num_of_child_label2 = right_children_label_dict[value_of_label2]
		string5 = (" " + str(node.right_children_value) + ": ")
		string6 = "[" + str(num_of_child_label1) + " " + str(value_of_label1) + \
			          " /" + str(num_of_child_label2) + " " + str(value_of_label2) + "]"
		print(new_str + string5 + string6)
		pretty_print(node.right, value_of_label1, value_of_label2)	


def make_prediction(attributes, observation, node):
	if node.vote != None:
		return node.vote
	else:
		split_attribute = node.split_attribute
		index_split_attribute = attributes.index(split_attribute)
		value = observation[index_split_attribute]
		if value == node.left_children_value:
			return make_prediction(attributes, observation, node.left)
		elif value == node.right_children_value:
			return make_prediction(attributes, observation, node.right)


def calc_error_rate_helper(prediction_lst, observations):
	total = len(observations)
	count = 0
	for i in range(len(prediction_lst)):
		if prediction_lst[i] != observations[i][-1]:
			count += 1
	return count / total 


def calculate_error_rate_for_classification(dataset, tree_model):
	attributes = dataset[0][:-1]
	observations = dataset[1:]
	prediction_lst = []
	for observation in observations:
		pred_label = make_prediction(attributes, observation, tree_model)
		prediction_lst.append(pred_label)
	error_rate = calc_error_rate_helper(prediction_lst, observations)
	return error_rate, prediction_lst


def train(dataset):
	max_depth = dataset.pop()
	attributes = dataset[0][:-1]
	observations = dataset[1:]
	current_depth = 0
	features_column = dataset[0][:-1]
	label_dict = create_label_dict(observations)
	value_of_label1 = None
	value_of_label2 = None
	for key in label_dict:
		if value_of_label1 == None:
			value_of_label1 = key
		else:
			value_of_label2 = key
	if max_depth == 0: # speciall case
		vote = majority_vote_classifier(observations)
		tree_model = DecisionTreeRootNode()
		tree_model.vote = vote
		tree_model.depth = 0
		tree_model.label_dict = label_dict
		pretty_print2(tree_model, value_of_label1, value_of_label2)
		return tree_model
	tree_model = build_tree(features_column, attributes, observations,
	                        max_depth, current_depth, 
	                        value_of_label1, value_of_label2) # maxdepth > 0
	pretty_print(tree_model, value_of_label1, value_of_label2)
	return tree_model


def write_prediction(prediction_lst, output):
	with open(output, 'w') as output_file:
		output_string = ""
		for label in prediction_lst:
			output_string += label + "\n"
		output_file.write(output_string.rstrip())


def write_metrics(train_error, test_error, metrics_out):
	with open(metrics_out, 'w') as f:
		f.write('error(train): ' + str(train_error) + '\n')
		f.write('error(test): ' + str(test_error))


def main(train_input, test_input, max_depth, train_out, test_out, metrics_out):
	dataset_train = Inport_data(train_input)
	dataset_train.append(max_depth)
	tree_model = train(dataset_train)
	train_error, prediction_lst_train = \
	calculate_error_rate_for_classification(dataset_train, tree_model)
	print("train_error :", train_error)
	write_prediction(prediction_lst_train, train_out)
	dataset_test = Inport_data(test_input)
	test_error, prediction_lst_test = \
	calculate_error_rate_for_classification(dataset_test, tree_model)
	print("test_error :", test_error)
	write_prediction(prediction_lst_test, test_out)
	write_metrics(train_error, test_error, metrics_out)


if __name__ == "__main__":
	train_input = sys.argv[1]
	test_input = sys.argv[2]
	max_depth = int(sys.argv[3])
	train_out = sys.argv[4]
	test_out = sys.argv[5]
	metrics_out = sys.argv[6]
	main(train_input, test_input, max_depth,
			train_out, test_out, metrics_out)




