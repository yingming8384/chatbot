import math
import os
import random
import getConfig
import jieba



### step 1 ###
### 读取配置 ###
gConfig = {}

gConfig=getConfig.get_config()

conv_path = gConfig['resource_data']

if not os.path.exists(conv_path):
	
	exit()
print('### step 1 ###')
print('### 配置读取完成 ###\n')


### step 2 ###
### 文本预处理 ###
convs = []  # 用于存储对话集合
# 下面的代码的含义是，将训练集中的每一段对话保存到 convs 变量中
# 也就是说，convs 变量的每个 item 是一段对话
with open(conv_path, encoding='utf-8') as f:
	one_conv = []        # 存储一次完整对话
	l = 0
	for line in f:
		line = line.strip('\n').replace('/', '')#去除换行符，去除原数据集中分割词语的'/'，后面用jieba分词
		# 如果这是一个空行，那么就处理下一行
		if line == '':
			continue
		# E 标签是用来分割每段话的
		# 如果遇到了 E，就说明一段对话已经结束，即将开始新的一段对话
		if line[0] == gConfig['e']:
			if one_conv:
				convs.append(one_conv)
			one_conv = []
		# M 表示这是一段信息
		# 如果遇到了 M，那么就将这段信息保存下来
		elif line[0] == gConfig['m']:
			# 第一个字符是 M，M 和信息之间是用空格分割的，所以 split 之后获取的是第一个元素而不是第零个元素
			one_conv.append(line.split(' ')[1])

		l+=1
		if l % 100000 == 0:
			print('step 2 convs 处理进度：%d' % l)
 
# 把对话分成问与答两个部分
ask = []        # 问
response = []   # 答
c = 0
for conv in convs:
	# 如果一段对话中只有一句话，那么说明没有问答，继续处理下一段对话
	if len(conv) == 1:
		continue
	# 因为默认是一问一答的，所以需要进行数据的粗裁剪，对话行数要是偶数的
	if len(conv) % 2 != 0:  
		conv = conv[:-1]
	for i in range(len(conv)):
		# 因为i是从0开始的，因此偶数行为发问的语句，奇数行为回答的语句
		# 用 jieba 分词，词语与词语之间用空格分割
		if i % 2 == 0:
			conv[i]=" ".join(jieba.cut(conv[i]))
			ask.append(conv[i])
		else:
			conv[i]=" ".join(jieba.cut(conv[i]))
			response.append(conv[i])
		if c%10000 == 0:
			print('step 2 问答处理进度：%d, %d' % (i, c))
	c += 1
print('### step 2 ###')
print('### 文本预处理完成 ###')
print('问的长度是', len(ask), '答的长度是', len(response), '\n')
 

### step 3 ###
### 创建训练集和测试集数据 ###
def convert_seq2seq_files(questions, answers, TESTSET_SIZE):
    # 创建文件
    train_enc = open(gConfig['train_enc'],'w', encoding='utf-8')  # 问
    train_dec = open(gConfig['train_dec'],'w', encoding='utf-8')  # 答
    test_enc  = open(gConfig['test_enc'], 'w', encoding='utf-8')  # 问
    test_dec  = open(gConfig['test_dec'], 'w', encoding='utf-8')  # 答
 
    
    # 随机生成测试集样本的编号
    test_index = random.sample([i for i in range(len(questions))],TESTSET_SIZE)
 
    # 生成训练集数据集合测试集数据
    for i in range(len(questions)):
        if i in test_index:
            test_enc.write(questions[i]+'\n')
            test_dec.write(answers[i]+ '\n' )
        else:
            train_enc.write(questions[i]+'\n')
            train_dec.write(answers[i]+ '\n' )
        if i % 10000 == 0:
            print(len(range(len(questions))), '处理进度：', i)
 
    train_enc.close()
    train_dec.close()
    test_enc.close()
    test_dec.close()
 
# 问的文件后缀是.enc，答的文件后缀是.dec
# 测试集数据从 10000 调整到 100000
convert_seq2seq_files(ask, response, 100000)
print('### step 3 ###')
print('### 训练集和测试集数据创建完成 ###')