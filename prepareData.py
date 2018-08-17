import math
import os
import random
import getConfig
from tensorflow.python.platform import gfile
import re
# 特殊标记，用来填充标记对话
PAD = "__PAD__"
GO = "__GO__"
# 对话结束
EOS = "__EOS__"  
# 标记未出现在词汇表中的字符
UNK = "__UNK__"  
START_VOCABULART = [PAD, GO, EOS, UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# 创建常用词汇表文件，文件中每一行是一个常用词汇
# input_file：对话文件，文件中每一行是一句话
# vocabulary_size：生成的常用词汇表的大小（不包含特殊字符），配置文件中设置为了 20000
# output_file：常用词汇表的保存路径
def create_vocabulary(input_file,vocabulary_size,output_file):
    vocabulary = {}
    k=int(vocabulary_size)
    with open(input_file,'r', encoding='utf-8') as f:
         counter = 0
         for line in f:
            counter += 1
            tokens = [word for word in line.strip().split()]
            # 对词语做词频统计
            for word in tokens:
                if word in vocabulary:
                   vocabulary[word] += 1
                else:
                   vocabulary[word] = 1
         
         # 获取文件中的词汇，将其按照出现频率从高到低的顺序排列，并且将特殊字符放在列表最前面
         vocabulary_list = START_VOCABULART + sorted(vocabulary, key=vocabulary.get, reverse=True)
          # 取前20000个常用汉字
         if len(vocabulary_list) > k:
            vocabulary_list = vocabulary_list[:k]
         print(input_file + " 词汇表大小:", len(vocabulary_list))
         # 将指定长度的常用词汇存放到指定文件中
         with open(output_file, 'w', encoding='utf-8') as ff:
               for word in vocabulary_list:
                   ff.write(word + "\n")

# 将 input_file 里的词语转换为对应的编号，并保存在output_file
def convert_to_vector(input_file, vocabulary_file, output_file):
	# 将字典文件中的常用词汇保存到列表 tmp_vocab 中
  tmp_vocab = []
  with open(vocabulary_file, "r", encoding='utf-8') as f:
    tmp_vocab.extend(f.readlines())
  tmp_vocab = [line.strip() for line in tmp_vocab]

	# 将词汇按照出现的顺序进行编号，从前面的处理可以知道，编号越小，词语出现的频率越高
  # 编号的结果保存到字典 vocab 中，key 是词语，value 是编号
  vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])

	# 将 input_file 里的词语转换为对应的编号，并保存在output_file
  output_f = open(output_file, 'w', encoding='utf-8')
  with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
      line_vec = []
      for word in line.strip().split():
        line_vec.append(vocab.get(word, UNK_ID))
      output_f.write(" ".join([str(num) for num in line_vec]) + "\n")
  output_f.close()


# 创建训练集和测试集对应的编号文件，创建词汇表文件，最后返回的是这些文件对应的路径
def prepare_custom_data(working_directory, 
  train_enc, train_dec, test_enc, test_dec, 
  enc_vocabulary_size, dec_vocabulary_size, 
  tokenizer=None):

    # Create vocabularies of the appropriate sizes.
    enc_vocab_path = os.path.join(working_directory, "vocab%d.enc" % enc_vocabulary_size)
    dec_vocab_path = os.path.join(working_directory, "vocab%d.dec" % dec_vocabulary_size)
    # 使用问训练集创建问词汇表文件
    create_vocabulary(train_enc,enc_vocabulary_size,enc_vocab_path)
    # 使用答训练集创建答词汇表文件
    create_vocabulary(train_dec,dec_vocabulary_size,dec_vocab_path)
   
    # 创建训练集数据的对应的编号文件
    enc_train_ids_path = train_enc + (".ids%d" % enc_vocabulary_size)
    dec_train_ids_path = train_dec + (".ids%d" % dec_vocabulary_size)
    # 使用 enc_vocab_path 词汇表创建 train_enc 对应的编号文件 enc_train_ids_path
    convert_to_vector(train_enc, enc_vocab_path, enc_train_ids_path)
    # 使用 dec_vocab_path 词汇表创建 train_dec 对应的编号文件 dec_train_ids_path
    convert_to_vector(train_dec, dec_vocab_path, dec_train_ids_path)
 

    # 创建测试集数据的对应的编号文件
    enc_dev_ids_path = test_enc + (".ids%d" % enc_vocabulary_size)
    dec_dev_ids_path = test_dec + (".ids%d" % dec_vocabulary_size)
    convert_to_vector(test_enc, enc_vocab_path, enc_dev_ids_path)
    convert_to_vector(test_dec, dec_vocab_path, dec_dev_ids_path)

    return (enc_train_ids_path, dec_train_ids_path, enc_dev_ids_path, dec_dev_ids_path, enc_vocab_path, dec_vocab_path)


# 用于语句切割的正则表达
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
  #将一个语句中的字符切割成一个list，这样是为了下一步进行向量化训练
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

def sentence_to_token_ids(sentence, vocabulary, normalize_digits=True):#将输入语句从中文字符转换成数字符号

  words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]

def initialize_vocabulary(vocabulary_path):#初始化字典，这里的操作与上面的48行的的作用是一样的，是对调字典中的key-value
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with open(vocabulary_path, "r", encoding='utf-8') as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)
