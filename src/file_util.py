## @package fio# Package for I/O# @file fio.py# @author Wencan Luo (wencan@cs.pitt.edu)# @date 2011-09-25import types as Typesimport sysimport jsonimport osimport codecsdef NewPath(path):	if not os.path.exists(path):		os.makedirs(path)def remove(file):	if IsExist(file):		os.remove(file)def IsExist(file):	return os.path.isfile(file)def IsExistPath(path):	return os.path.exists(path)def DeleteFolder(path):	try:		shutil.rmtree(path)	except Exception:		pass# 	for dirpath, dirnames, filenames in os.walk(path):# 		try:# 			os.rmdir(dirpath)# 		except OSError as ex:# 			print(ex)def ReadFile(file):	"""Input a file, and return a list of sentences		@param file: string, the input file path	@return: list of lines. Note: each line ends with "\r\n" or "\n"	"""		#read the file	f = open(file,'r')	lines = f.readlines()	f.close()		return lines	def ReadMatrix(file, hasHead=True):	"""	Function: Load a matrix from a file. The matrix is M*N	@param file: string, filename	@param hasHead: bool, whether the file has a header	"""	lines = ReadFile(file)	#print len(lines)		tm = []	for line in lines:		row = []		line = line.strip()		if len(line) == 0: continue		for num in line.split("\t"):			row.append(num.strip())		tm.append(row)	if hasHead:		header = tm[0]		body = tm[1:]		return header, body	else:		return tm      def WriteMatrix(file, data, header=None):	"""	Function: save a matrix to a file. The matrix is M*N	@param file: string, filename	@param data: M*N matrix,  	@param header: list, the header of the matrix	"""	reload(sys)	sys.setdefaultencoding('utf8')    	SavedStdOut = sys.stdout	sys.stdout = open(file, 'w')		if header != None:		for j in range(len(header)):			label = header[j]			if j == len(header)-1:				print(label)			else:				sys.stdout.write(label+"\t")	for row in data:		for j in range(len(row)):			col = row[j]			if j == len(row) - 1:				print(col)			else:				sys.stdout.write(str(col)+"\t")		sys.stdout = SavedStdOut      def LoadDict(file, type='str'):	"""	@function:load a dict	@return dict: dictionary	"""	body = ReadMatrix(file, False)		if body == None: return None	dict = {}	for row in body:		assert(len(row) == 2)				if type == 'str' or type==str:			dict[row[0]] = row[1]		if type == 'float' or type == float:			dict[row[0]] = float(row[1])		if type == 'int' or type == int:			dict[row[0]] = int(row[1])				return dictdef SaveDict(dict, file, SortbyValueflag = False):	"""	@function:save a dict	@param dict: dictionary	"""	SavedStdOut = sys.stdout	sys.stdout = open(file, 'w')		if SortbyValueflag:		for key in sorted(dict, key = dict.get, reverse = True):			print(str(key) + "\t" + str(dict[key]))	else:		for key in sorted(dict.keys()):			print(str(key) + "\t" + str(dict[key]))	sys.stdout = SavedStdOutdef SaveDict2Json(dict, file):	with codecs.open(file, "w", 'utf-8') as fout:		json.dump(dict, fout, indent=2)def LoadDictJson(file):	with open(file, "r") as fin:		dict = json.load(fin)	return dict	if __name__ == '__main__':	pass	