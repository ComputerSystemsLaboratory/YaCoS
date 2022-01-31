"""
Copyright 2021 Anderson Faustino da Silva.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import csv
import glob
import tempfile
import subprocess
import yaml

import math
import numpy as np



from absl import logging as lg

from yacos.essential import IO
from yacos.essential import Engine
from yacos.essential import Sequence

from PIL import Image
from skimage.feature import local_binary_pattern
#from cython_rbp import _cython_rbp



class bit2vec:
	@property
	def array(self):
		return self._array
	
	@property
	def columns(self):
		return self._columns

	@columns.setter
	def columns(self,c):
		self._columns = c

	@property
	def desired_lines(self):
		return self._desired_lines

	@desired_lines.setter
	def desired_lines(self,l):
		self._desired_lines = l

	@staticmethod
	def get_array(binary_name):
		#recieves a binary file and turns it into a array of 0's and 1's
		f=open(binary_name,'rb')
		binary_data = f.read()
		f.close()
		linear_data=[]
		for bt in binary_data:
			linear_data.append(bt)
		del linear_data[-1]#remove eof char

		byte_array = np.array(linear_data,
							  dtype=np.uint8)
		bits_array = np.unpackbits(byte_array)
		return bits_array

	def __init__(self,
				 bits_array,
				 desired_lines=0,
				 columns=0):
		np_bits_array = np.array(bits_array,
								 dtype=np.uint8)
		self._array = np_bits_array
		self._desired_lines = desired_lines
		self._columns = columns

	def __calc_num_add_lines(self):
		num_add_lines = 0
		nlines = math.ceil(len(self._array)/self._columns)
		if nlines < self._desired_lines:
			num_add_lines = self._desired_lines-nlines
		return num_add_lines

	def __calc_nlines(self):
		nlines = math.ceil(len(self._array)/self._columns)
		if nlines < self._desired_lines:	
			nlines = self._desired_lines
		return nlines

	def __calc_missing_bits(self,
							num_add_lines):
		missing = self._columns*num_add_lines
		last_column_bits = len(self._array)%self._columns
		if last_column_bits != 0:
			missing += self._columns - (last_column_bits)
		return missing

	def create_npz(self,
				   output_file=None,
				   fill_np=-1):
		bits_array = self._array
		
		nlines = self.__calc_nlines()
		num_add_lines = self.__calc_num_add_lines()
		missing = self.__calc_missing_bits(num_add_lines)
		np_bits_array = np.array(bits_array,dtype=np.double)
		np_missing_array = np.array([fill_np]*missing,dtype=np.double)
		np_bits_array = np.append(np_bits_array, np_missing_array)
		np_matrix = np_bits_array.reshape(nlines,self._columns)
		if output_file != None:
			np.savez_compressed(output_file,  
								values=np_matrix)
		return np_matrix

	def create_img(self,
				   output_img_file=None,
				   fill_img=128):
		bits_array = self._array

		nlines = self.__calc_nlines()
		num_add_lines = self.__calc_num_add_lines()
		missing = self.__calc_missing_bits(num_add_lines) 

		img_missing_array = np.array([fill_img]*missing,dtype=np.uint8)
		img_bits_array = bits_array*255
		img_bits_array = np.append(img_bits_array,img_missing_array)
		img_matrix = img_bits_array.reshape(nlines,self._columns)

		if output_img_file != None:
			img = Image.fromarray(img_matrix,'L')
			img.save(output_img_file)
		return img_matrix

	def vertical_fold(self,
					  fold_lines,
					  output_file=None,
					  fill_value=0):
		bits_array = self._array
		
		nlines = math.ceil(len(self._array)/self._columns)
		fold_count = math.ceil(nlines/fold_lines)
		total_lines = fold_count * fold_lines

		bits_array = np.array(bits_array, dtype=np.double)
		missing = total_lines*self._columns - len(bits_array)
		missing_array = np.array([fill_value]*missing, dtype=np.double)
		bits_array = np.append(bits_array, missing_array)
		matrix = bits_array.reshape(total_lines,self._columns)
		fold_matrix = np.array(matrix[0:fold_lines])

		exp = 1
		for i in range(fold_lines,matrix.shape[0]):
			if i%fold_lines == 0:
				exp /= 2.0
				if exp == 0:
					print('Exponent reached 0. '+
						  'There will be information loss in file:{0}'
						  .format(output_file),
						  file=sys.stderr)
			for j in range(matrix.shape[1]):
				fold_matrix[i%fold_lines, j] += matrix[i,j] * exp

		if output_file != None:
			np.savez_compressed(output_file,values=fold_matrix)

		return fold_matrix

	def create_normalized_LBP_histogram(self, 
										P=8, 
										R=2, 
										method='default', 
										output_file=None, 
										fill_value=0):
	
		img = self.create_npz(fill_np=fill_value)
		lbp = local_binary_pattern(img,P,R,method=method)
		max_val = int(2**P)
		histogram = np.zeros(max_val,dtype=np.double)
		for val in lbp.ravel():
			histogram[int(val)]+=1
		sum_hist = np.sum(histogram)
		norm_hist = histogram/sum_hist
		if output_file != None:
			np.savez_compressed(output_file,values=norm_hist)
		return norm_hist

	def create_normalized_RBP_histogram(self,
										P=8,
										R=2,
										output_file=None):
		#raw binary patter
		#an adaptation of LBP to binary images
		
		def __get_pixel(img,row, column):
			r = np.int64(np.round(row))
			c = np.int64(np.round(column))

			if (r<0) or (c<0) or (r >= img.shape[0]) or (c >= img.shape[1]):
				return 0
			else:
				return img[r,c]
		def __raw_binary_pattern(img, P=8, R=2):
			
		#		This function extracts the binary pattern around each "pixel of
		#		the binary image. The strategy is very similar to LBP, but
		#		instead of compare the pixel with their neighbors, it just take
		#		the pattern looking at its neighbors.
		#		Exemple:
		#		Supose this part of an BINARY image and the pixel X. 
		#		0   1   1
		#		1	X	1			
		#		0	0	1
		#	
		#		Consider P=8 and R=1
		#
		#		Pattern around X is 11001011.
		#
		#		>-------V
		#		:		:
		#		: pixel	: <---pattern start here
		#		:		:
		#		^-------<


			rr = -R*np.sin(2*np.pi*np.arange(P,dtype=np.double)/P)
			rr = np.round(rr,5)

			cc = R*np.cos(2*np.pi*np.arange(P,dtype=np.double)/P)
			cc = np.round(cc,5)

			rbp_matrix=np.zeros((img.shape[0],img.shape[1]),dtype=np.double)
			r=np.int64(0)
			c=np.int64(0)
			pattern = np.int64(0)
			p=np.int64(0)
			m=np.int64(0)
			while r < img.shape[0]:
				c=0
				while c < img.shape[1]:
					pattern = 0
					p=0
					m=1
					while p < P:
						pixel = __get_pixel(img,r+rr[p],c+cc[p])
						pattern += pixel * m
						m = m*2
						p = p+1
					rbp_matrix[r,c]=pattern
					c+=1
				r+=1
			return rbp_matrix
		
		img = self.create_npz(fill_np=0.0)
		#rbp = _cython_rbp(img,P=P,R=R) #see cython_rbp.pyx file
		rbp = __raw_binary_pattern(img,P=P,R=R)
		max_val = int(2**P)
		histogram = np.zeros(max_val,dtype=np.double)
		for val in rbp.ravel():
			histogram[int(val)]+=1
		sum_hist = np.sum(histogram)
		norm_hist = histogram/sum_hist
		if output_file != None:
			np.savez_compressed(output_file,values=norm_hist)
		return norm_hist


class Prog2Image:
    """Prog2Image Representation."""

    __version__ = '2.0.0'

    @staticmethod
    def extract_from_binary(binary_name,
	                        columns=256,
                            lines=0):
        """Extract prog2image representation form a binary file
        Parameters
        ----------
        binary_name: str
        columns: int
        lines: int 
            if line is lower than the size/columns (size give in bits)
            the number of lines will be setted automatically to store all bits

        Return
        ------
        representation : np.array
        """

        benchmark_array = bit2vec.get_array(binary_name)
        benchmark_emb = bit2vec(benchmark_array,
                                columns=columns,
                                desired_lines=lines)
        representation = benchmark_emb.create_npz()
        return representation



    @staticmethod
    def compile_and_extract(benchmarks_base_directory,
                            benchmarks_filename,
                            sequence='-O0',
                            columns=256,
                            lines=0):
        """Compile the benchmark and extract prog2image representation.
            The benchmark directory must have a Makefile.opt that generates
            the bytecode as a.out_o.bc
        Parameters
        ----------
        benchmarks_base_directory: str

        benchmarks_filename: str

        sequence : list

        Return
        ------
        processed : dict {benchmark: embeddings}
        """

        processed = {}
        benchmarks = IO.load_yaml_or_fail(benchmarks_filename)

        for bench in benchmarks:
            idx = bench.find('.')
            collection = bench[:idx]
            benchmark = bench[idx+1:]
            benchmark_dir = os.path.join(benchmarks_base_directory,
                                         collection,
                                         benchmark)
            Engine.compile(benchmark_dir,'opt',sequence)
            bytecode_file = os.path.join(benchmark_dir,'a.out_o.bc')
            #benchmark_array = bit2vec.get_array(bytecode_file)
            #benchmark_emb = bit2vec(benchmark_array,
            #						columns=columns,
            #						desired_lines=lines)
            benchmark_emb = Prog2Image.extract_from_binary(bytecode_file,
                                                           columns=columns,
                                                           lines=lines)
            processed[benchmark] = benchmark_emb
    
        return processed

    @staticmethod
    def extract(benchmarks_base_directory,
                benchmarks_filename,
                columns=256,
                lines=0):
        """Extract prog2image representation.

        Parameters
        ----------
        benchmarks_filename: str

        Return
        ------
        processed : dict {benchmark: embeddings}
        """

        processed = {}
        benchmarks = IO.load_yaml_or_fail(benchmarks_filename)
        for bench in benchmarks:
            idx = bench.find('.')
            collection = bench[:idx]
            benchmark = bench[idx+1:]
            benchmark_dir = os.path.join(benchmarks_base_directory,
                                         collection,
                                         benchmark)
            bytecode_file = os.path.join(benchmark_dir,'a.out_o.bc')
            benchmark_emb = Prog2Image.extract_from_binary(bytecode_file,
                                                           columns=columns,
                                                           lines=lines)
            processed[benchmark] = benchmark_emb
        return processed

		

class LBPeq:
    """LBPeq Representation."""

    __version__ = '2.0.0'

    @staticmethod 
    def extract_from_binary(binary_name,
                            columns=256,
                            lines=0):
        benchmark_array = bit2vec.get_array(binary_name)
        benchmark_emb = bit2vec(benchmark_array,
                                columns=columns,
                                desired_lines=lines)
        representation = benchmark_emb.create_normalized_RBP_histogram()
        return representation


    @staticmethod
    def compile_and_extract(benchmarks_base_directory,
                            benchmarks_filename,
                            sequence='-O0',
                            columns=256,
                            lines=0):
        """Compile the benchmark and extract prog2image representation.

        Parameters
        ----------
        benchmarks_filename: str

        sequence : list

        Return
        ------
        processed : dict {benchmark: embeddings}
        """
        processed = {}
        benchmarks = IO.load_yaml_or_fail(benchmarks_filename)

        for bench in benchmarks:
            idx = bench.find('.')
            collection = bench[:idx]
            benchmark = bench[idx+1:]
            benchmark_dir = os.path.join(benchmarks_base_directory,
                                         collection,
                                         benchmark)
            Engine.compile(benchmark_dir,'opt',sequence)
            bytecode_file = os.path.join(benchmark_dir,'a.out_o.bc')
            benchmark_emb = LBPeq.extract_from_binary(bytecode_file,
                                                      columns=columns,
                                                      lines=lines)
            processed[benchmark] = benchmark_emb
    
        return processed

    @staticmethod
    def extract(benchmarks_base_directory,
                benchmarks_filename,
                columns=256,
                lines=0):
        """Extract prog2image representation.

        Parameters
        ----------
        benchmarks_filename: str

        Return
        ------
        processed : dict {benchmark: embeddings}
        """
        processed = {}
        benchmarks = IO.load_yaml_or_fail(benchmarks_filename)
        for bench in benchmarks:
            idx = bench.find('.')
            collection = bench[:idx]
            benchmark = bench[idx+1:]
            benchmark_dir = os.path.join(benchmarks_base_directory,
                                         collection,
                                         benchmark)
            bytecode_file = os.path.join(benchmark_dir,'a.out_o.bc')
            benchmark_emb = LBPeq.extract_from_binary(bytecode_file,
                                                      columns=columns,
                                                      lines=lines)
            processed[benchmark] = benchmark_emb
        return processed

class LBPif:
    """LBPeq Representation."""

    __version__ = '2.0.0'

    @staticmethod
    def extract_from_binary(binary_name,
                            columns=256,
                            lines=0):
        benchmark_array = bit2vec.get_array(binary_name)
        benchmark_emb = bit2vec(benchmark_array,
                                columns=columns,
                                desired_lines=lines)
        representation = benchmark_emb.create_normalized_LBP_histogram()
        return representation

    @staticmethod
    def compile_and_extract(benchmarks_base_directory,
                            benchmarks_filename,
                            sequence='-O0',
                            columns=256,
                            lines=0):
        """Compile the benchmark and extract prog2image representation.

        Parameters
        ----------
        benchmarks_filename: str

        sequence : list

        Return
        ------
        processed : dict {benchmark: embeddings}
        """
        processed = {}
        benchmarks = IO.load_yaml_or_fail(benchmarks_filename)

        for bench in benchmarks:
            idx = bench.find('.')
            collection = bench[:idx]
            benchmark = bench[idx+1:]
            benchmark_dir = os.path.join(benchmarks_base_directory,
                                         collection,
                                         benchmark)
            Engine.compile(benchmark_dir,'opt',sequence)
            bytecode_file = os.path.join(benchmark_dir,'a.out_o.bc')
            benchmark_emb = LBPif.extract_from_binary(bytecode_file,
                                                      columns=columns,
                                                      lines=lines)
            processed[benchmark] = benchmark_emb
    
        return processed

    @staticmethod
    def extract(benchmarks_base_directory,
                benchmarks_filename,
                columns=256,
                lines=0):
        """Extract prog2image representation.

        Parameters
        ----------
        benchmarks_filename: str

        Return
        ------
        processed : dict {benchmark: embeddings}
        """
        processed = {}
        benchmarks = IO.load_yaml_or_fail(benchmarks_filename)
        for bench in benchmarks:
            idx = bench.find('.')
            collection = bench[:idx]
            benchmark = bench[idx+1:]
            benchmark_dir = os.path.join(benchmarks_base_directory,
                                         collection,
                                         benchmark)
            bytecode_file = os.path.join(benchmark_dir,'a.out_o.bc')
            benchmark_emb = LBPif.extract_from_binary(bytecode_file,
                                                      columns=columns,
                                                      lines=lines)
            processed[benchmark] = benchmark_emb
        return processed
