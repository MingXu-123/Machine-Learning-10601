import os
import sys

arg1 = sys.argv[1]
arg2 = sys.argv[2]


def isEmptyFile(file):
	if os.stat(file).st_size == 0:
		return True

def readFile(path):
	with open(path, "r") as f:
		return f.readlines()

def writeFile(path, contents):
	with open(path, "w") as f:
		f.write(contents)

def main():
	if isEmptyFile(arg1):
		open(arg2, "w").close()
	inputfile = readFile(arg1)
	outputfileList = []
	for i in range(len(inputfile) - 1, -1, -1):
		outputfileList.append(inputfile[i])
	res = ""
	for i in range(len(outputfileList)):
		res += outputfileList[i]
	writeFile(arg2, res)

if __name__ == '__main__':
	main()

