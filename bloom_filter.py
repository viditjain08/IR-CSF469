import os
import sys
import mmh3
import json
import math
import BitVector

from rabin import rabin as rabin_fingerprint_function


class BloomFilter(object):
	def __init__(self, m, k):
		self.m = m # number of bits in the array
		self.k = k # number of hashes to be used
		self.n = 0
		self.bv = BitVector.BitVector(size = self.m)
		for i in self.bv:
			self.bv[i] = 0

	def getBitArrayIndices(self, key):
		returnList = []
		for i in range(1, self.k + 1):
			returnList.append((hash(key) + i * mmh3.hash(key)) % self.m)
		return returnList

	def add(self, key):
		for i in self.getBitArrayIndices(key):
			self.bv[i] = 1
		self.n += 1

	def compare(self, comparator):
		if abs(self.n - comparator.n) < 30:
			count = 0
			for x in range(self.m):
				if self.bv[x] == comparator.bv[x]:
					count += 1
			if count / self.m > 0.65:
				return True
		return False



def rabin_fingerprint(file_path):
	file_len = len(open(file_path, 'r').read())
	chunk_indexes = [(x * file_len) // 25 for x in range(25)] # Initialising chunk indexes

	chnuk_indexes = rabin_fingerprint_function(file_path, 32 * 1024, 0, 64 * 1024, 3, 48)
	return chunk_indexes


def bloom_filter(text_chunks):
	filter = BloomFilter(64, int(64 * math.log(2)/ len(text_chunks)))
	for chunk in text_chunks:
		filter.add(chunk)
	return filter
	


def bloom_file(file_path):
	chunk_indexes = rabin_fingerprint(file_path)
	file = open(file_path, 'r')
	file_text = file.read()
	file_text_chunks = []
	for x in range(1, len(chunk_indexes)):
		file_text_chunks.append(file_text[chunk_indexes[x - 1] : chunk_indexes[x]])
	
	return bloom_filter(file_text_chunks)
		

def read_dir(path):
    files = os.listdir(path)
    dirs = []
    for file_name in files:
        with open(os.path.join(path, file_name)) as f:
        	dirs.append(f.read())
    return dirs


if __name__ == '__main__':
	dirs_text = sys.argv[1]
	bloom_arr = []
	for file_text in os.listdir(dirs_text):
		bloom_arr.append(bloom_file(os.path.join(dirs_text, file_text)))
	with open('bloom.txt', 'w') as f:
		string = ''
		for x in bloom_arr:
			string += "".join(list(map(str,x.bv))) + '\n'
		f.write("\n".join(string))
		for i in range(len(bloom_arr)):
			for j in range(i):
				if bloom_arr[i].compare(bloom_arr[j]):
					print(True)
