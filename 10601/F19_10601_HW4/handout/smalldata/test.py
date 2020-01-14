def parse_formatted_data(input_file):
	line_lst = None
	with open(input_file, 'r') as f:
		lines = f.readlines()
		print(lines)
		# reader = csv.reader(f, delimiter='\t')
		# for row in reader:
		# 	print(row)
		# 	print(len(row))
		# 	break