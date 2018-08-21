import math
import os
import random
import sys
import time
import prepareData

import numpy as np
from six.moves import xrange
import tensorflow as tf

from configparser import SafeConfigParser
import seq2seq_model
    
gConfig = {}

def get_config(config_file='seq2seq.ini'):
    parser = SafeConfigParser()
    parser.read(config_file)
    # get the ints, floats and strings
    _conf_ints = [ (key, int(value)) for key,value in parser.items('ints') ]
    _conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
    _conf_strings = [ (key, str(value)) for key,value in parser.items('strings') ]
    return dict(_conf_ints + _conf_floats + _conf_strings)

# 设置不同的桶以及桶的长度，原则两个：1、尽量覆盖所有训练语料语句的长度，2、尽量保持桶里语料的平衡
# 下面定义了一个列表，每个元素的第一个值表示问的长度的限制，第二个值表示答的长度的限制
# 在构建数据集的时候，不能超过这个限制
_buckets = [(1, 10), (10, 15), (20, 25), (40, 50)]


# 根据桶的设置返回对应的训练集和测试集数据
def read_data(source_path, target_path, max_size=None):
  """
  从 source_path 和 target_path 中读取数据并且放到 buckets 中。

  参数:
    source_path: 问编号文件的路径。
    target_path: 答编号文件的路径，它必须和 source_path 对应，也就是说一问一答。
    max_size: 最多读入多少行数据。如果设置为了 0 或者 None，那么就会读取所有的数据。

  返回:
    data_set：包含的是与 _buckets 中不同的桶的设置对应的数据集（训练集和测试集）
  """
  data_set = [[] for _ in _buckets]
  # 我们可以使用 open 打开文件，但是 GFile 更能适应各种不同的文件系统，比如 Google cloud system，HDFS
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        # 在目标末尾加上结束字符
        target_ids.append(prepareData.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
        	# 只有当样本的长度满足要求的时候，才会将这个样本添加到 data_set 中
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


# session 用于分配资源，要想执行 tensorflow 中的运算，就要创建 session 为其分配资源
def create_model(session, forward_only):

  """Create model and initialize or load parameters"""
  model = seq2seq_model.Seq2SeqModel( gConfig['enc_vocab_size'], 
                                      gConfig['dec_vocab_size'], 
                                      _buckets, gConfig['layer_size'], 
                                      gConfig['num_layers'], 
                                      gConfig['max_gradient_norm'], 
                                      gConfig['batch_size'], 
                                      gConfig['learning_rate'], 
                                      gConfig['learning_rate_decay_factor'], 
                                      forward_only=forward_only)

  # 如果配置文件中有预训练好的模型，那么就加载这个模型
  if 'pretrained_model' in gConfig:
      model.saver.restore(session,gConfig['pretrained_model'])
      return model

  # 获取 checkpoint 的状态
  ckpt = tf.train.get_checkpoint_state(gConfig['working_directory'])
  # 如果做过 checkpoint 操作，那么就读取保存的参数
  # 如果没有，就初始化所有变量
  if ckpt and ckpt.model_checkpoint_path:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model


def train():
 # 准备数据
  print("准备 %s 目录中的数据" % gConfig['working_directory'])
  # 创建训练集和测试集的对应的编号文件（将词语转换为编号），创建词汇表文件，下面返回的是这些文件对应的文件路径
  enc_train, dec_train, enc_test, dec_test, _, _ = prepareData.prepare_custom_data(gConfig['working_directory'],
    gConfig['train_enc'],
    gConfig['train_dec'],
    gConfig['test_enc'],
    gConfig['test_dec'],
    gConfig['enc_vocab_size'],
    gConfig['dec_vocab_size'])

 
  # setup config to use BFC allocator
  # config = tf.ConfigProto()  
  # config.gpu_options.allocator_type = 'BFC'

  # with tf.Session(config=config) as sess:
  with tf.Session() as sess:
    # Create model.
    print("创建 %d 层模型，每层有 %d 个单元." % (gConfig['num_layers'], gConfig['layer_size']))
    model = create_model(sess, False)
    print("### 模型创建完成！ ###")

    # 读取数据并计算其长度，将其放入对应的桶中
    print ("读取训练集数据和测试集数据 (训练集大小限制: %d)."
           % gConfig['max_train_data_size'])
    test_set = read_data(enc_test, dec_test)
    train_set = read_data(enc_train, dec_train, gConfig['max_train_data_size'])
    # 统计每个桶对应的训练集大小
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    print('每个桶对应的训练集大小: ', train_bucket_sizes)
    # 统计训练集总大小
    train_total_size = float(sum(train_bucket_sizes))
    print('训练集总大小: ', train_total_size)


    # 计算每个桶的训练集的数量在总数据集中的占比，结果累加
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # 开始循环训练
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    # 模型会一直训练下去，需要我们自己手动停止，但是每隔固定的 step 就会自动保存，所以训练成果会被保存下来
    while True:
      # 从 0 到 1 之间随机选择一个数，根据上面计算出来的占比选择样本
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time)
      loss += step_loss
      current_step += 1

      # 每隔 steps_per_checkpoint, 我们都会将训练成果保存，打印训练结果
      # 并且重置 step_time 和 loss
      if current_step % gConfig['steps_per_checkpoint'] == 0:
        perplexity = math.exp(loss/gConfig['steps_per_checkpoint']) if (loss/gConfig['steps_per_checkpoint']) < 300 else float('inf')
        print ("global step %d learning rate %.4f average step-time %.2f average perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time/gConfig['steps_per_checkpoint'], perplexity))
        # 如果当前 loss 比过去 3 次训练的 loss 都大，我们就降低学习速率
        if len(previous_losses) > 2 and step_loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        # 保存当前 loss
        previous_losses.append(step_loss)
        # 设置保存路径Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(gConfig['working_directory'], "seq2seq.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        # 重置 step_time 和 loss
        step_time, loss = 0.0, 0.0
        # 在测试集上测试模型的性能
        for bucket_id in xrange(len(_buckets)):
          # 如果测试集中没有数据就继续下次循环
          if len(test_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              test_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()

def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


def init_session(sess, conf='seq2seq.ini'):
    global gConfig
    gConfig = get_config(conf)
 
    model = create_model(sess, True)
    # 一问一答
    model.batch_size = 1 

    # 获取词典路径
    enc_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.enc" % gConfig['enc_vocab_size'])
    dec_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.dec" % gConfig['dec_vocab_size'])

    # 问是字典类型的对象
    enc_vocab, _ = prepareData.initialize_vocabulary(enc_vocab_path)
    # 答是列表类型的对象
    _, rev_dec_vocab = prepareData.initialize_vocabulary(dec_vocab_path)

    return sess, model, enc_vocab, rev_dec_vocab

# 一问一答
def decode_line(sess, model, enc_vocab, rev_dec_vocab, sentence):
    # 把句子转换成编号列表
    # tf.compat.as_bytes() 用于将句子用 utf-8 编码而不管传进来的句子是已经编码好的还是没有编码好的
    token_ids = prepareData.sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)

    # 判断问属于哪个 bucket，取最小的
    bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])

    # 对句子编号进行处理，产生模型的正确输入
    encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, 
                                                                      bucket_id)

    # 使用模型进行预测
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

    # 将输出转换成词语的数字编号
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

    # 如果生成了 EOS，那么就将 EOS 后面的部分全部裁减掉
    if prepareData.EOS_ID in outputs:
        outputs = outputs[:outputs.index(prepareData.EOS_ID)]

    return " ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs])

if __name__ == '__main__':
    # 可以从命令行中传入配置文件的路径
    if len(sys.argv) - 1:
        gConfig = get_config(sys.argv[1])
    # 如果没有设置配置文件的路径，使用默认参数中的配置文件的路径，也就是 seq2seq.ini 文件
    else:
        gConfig = get_config()

    print('\n>> Mode : %s\n' %(gConfig['mode']))

    if gConfig['mode'] == 'train':
        train()
    elif gConfig['mode'] == 'server':
    
        print('Serve Usage : >> python3 webui/app.py')
        print('# 使用 seq2seq_serve.ini 作为配置文件')