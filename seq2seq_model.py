# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

import random

import numpy as np
from six.moves import xrange
import tensorflow as tf

import prepareData


class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self, source_vocab_size, target_vocab_size, buckets, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, use_lstm=False,
               num_samples=512, forward_only=False):
    """Create the model.

    Args:
      source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. I 其实就是问，O 其实就是答。如果长度大于当前 bucket 设置，那么会传入
        下一个 bucket，并且自动进行 padding。
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: 模型每一层的单元的数量。
      num_layers: number of layers in the model.
      max_gradient_norm: 对梯度进行裁剪，用于防止梯度爆炸。
      batch_size
      learning_rate
      learning_rate_decay_factor
      use_lstm: 如果设置为 true，那么就使用 LSTM，否则使用 GRU。
      num_samples: number of samples for sampled softmax.
      forward_only: 如果为 True，那么不构建模型的后向传播。
    """
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None

    if num_samples > 0 and num_samples < self.target_vocab_size:
      # tf.get_variable 相比于 tf.Variable 更易于共享变量
      # 由于本项目中每层网络的单元的数量是一样的，所以 w 总共有 size * target_vocab_size 个参数
      # 因此下面的 w 是整个网络的权重，如果我们每一层的单元不一样，就要为不同的层定义不同的 w
      w = tf.get_variable("proj_w", [size, self.target_vocab_size])
      w_t = tf.transpose(w)
      b = tf.get_variable("proj_b", [self.target_vocab_size])
      output_projection = (w, b)

      # 定义 sampled_loss
      def sampled_loss(labels,logits):
        labels = tf.reshape(labels, [-1, 1])
        return tf.nn.sampled_softmax_loss(w_t, b, labels,logits,
                                          num_samples,
                                          self.target_vocab_size)
      
      # 将网络的损失函数设置为前面设置的 sampled_loss
      softmax_loss_function = sampled_loss
      
    # 下面三行代码用于解决深拷贝带来的问题，必须要有，不必深究
    setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
    setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
    setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)

    # 默认创建具有 size 个隐藏单元的 GRU 层
    single_cell = tf.contrib.rnn.GRUCell(size)
    # 可以设置使用 LSTM
    if use_lstm:
      # 创建具有 size 个隐藏单元的 LSTM 层
      single_cell = tf.contrib.rnn.BasicLSTMCell(size)
    cell = single_cell
    if num_layers > 1:
      # 创建多层循环神经网络
      cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers)

    # 下面的函数定义了一个序列到序列模型，RNN+Attention
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
      # encoder_inputs 和 decoder_inputs 的 shape 都是 [batch_size]
      # 下面的函数为 encoder_inputs 创建一个嵌入层，使用 RNN 对其进行编码，将编码结果保留给后面的 Attention 使用
      # 也会为 decoder_inputs 创建一个嵌入层，使用 RNN 对其进行编码，
      # 之后会运行 Attention 解码器，它会使用到前面产生的信息
      # 标准情况下，我们都会将 feed_previous 参数设置为 False
      # 该函数的返回值是 (outputs, state)
      # 其中的 outputs 在这里就是 [batch_size * target_vocab_size]
      return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
          encoder_inputs, decoder_inputs, cell,
          num_encoder_symbols=source_vocab_size,
          num_decoder_symbols=target_vocab_size,
          embedding_size=size,
          output_projection=output_projection,
          feed_previous=do_decode)

    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    # buckets[-1][0] 是 bucket 中定义的最大的问的长度
    for i in xrange(buckets[-1][0]):
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    # buckets[-1][1] 是 bucket 中定义的最大的答的长度
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))

    # targets 是 decoder_inputs 向左偏移一个 item.
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]

    # Training outputs and losses.
    if forward_only:
      self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
          softmax_loss_function=softmax_loss_function)
      # If we use output projection, we need to project outputs for decoding.
      if output_projection is not None:
        for b in xrange(len(buckets)):
          self.outputs[b] = [
              tf.matmul(output, output_projection[0]) + output_projection[1]
              for output in self.outputs[b]
          ]
    else:
      # 创建一个支持 bucket 的序列到序列模型
      self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, False),
          softmax_loss_function=softmax_loss_function)

    # 使用梯度下降优化模型
    # 返回所有 trainable=True 的 Variable
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in xrange(len(buckets)):
        # 返回梯度列表
        gradients = tf.gradients(self.losses[b], params)
        # 返回裁剪过的梯度和全局 norm（全局 norm 的计算参见文档）
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_gradient_norm)
        self.gradient_norms.append(norm)
        # 将梯度和参数进行一一对应，设置了 global_step 之后，每次操作都会增加 global_step 的值
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

    # 保存所有的 Variable，便于之后恢复
    self.saver = tf.train.Saver(tf.all_variables())

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

  def get_batch(self, data, bucket_id):
    """从指定的 bucket 中获取一批样本用于训练。

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      # 从对应的桶中随机选择一个样本
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # padding 长度不够的 encoder_inputs
      encoder_pad = [prepareData.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # 先添加一个 GO 符号到 decoder_inputs 中，然后 padding
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([prepareData.GO_ID] + decoder_input +
                            [prepareData.PAD_ID] * decoder_pad_size)

    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # batch_encoder_inputs 和 encoder_inputs 的内容是一样的，不过 re-index 罢了
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # batch_decoder_inputs 和 decoder_inputs 的内容是一样的，不过 re-index 罢了
    # 同时创建 batch_weights
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # 创建 batch_weights 矩阵，初始化为全 1 的矩阵
      # 填充部分的权重设置为 0
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # target 是 decoder_input 向左偏移一个 item 的结果
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == prepareData.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights
