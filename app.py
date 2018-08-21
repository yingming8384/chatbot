from flask import Flask, render_template, request, jsonify
import time
import threading
import jieba
import re
import tensorflow as tf
import execute

# 下面的函数没有太大的作用
# def heartbeat():
#     print (time.strftime('%Y-%m-%d %H:%M:%S - heartbeat', time.localtime(time.time())))
#     timer = threading.Timer(60, heartbeat)
#     timer.start()
# timer = threading.Timer(60, heartbeat)
# timer.start()


zhPattern = re.compile(u'[\u4e00-\u9fa5]+')

app = Flask(__name__,static_url_path="/static")

@app.route('/message', methods=['POST'])
def reply():

    req_msg = request.form['msg']
    res_msg = '^_^'
    req_msg=" ".join(jieba.cut(req_msg))
    #print(req_msg)
    res_msg = execute.decode_line(sess, model, enc_vocab, rev_dec_vocab, req_msg )
    
    res_msg = res_msg.replace('_UNK', '^_^')
    res_msg=res_msg.strip()
    
    # 如果接受到的内容为空，则给出相应的恢复
    if res_msg == ' ':
      res_msg = '请与我聊聊天吧'

    return jsonify( { 'text': res_msg } )

@app.route("/")
def index(): 
    return render_template("index.html")


sess = tf.Session()
sess, model, enc_vocab, rev_dec_vocab = execute.init_session(sess, conf='seq2seq_serve.ini')


# 启动APP
if (__name__ == "__main__"): 
    app.run(host = '0.0.0.0', port = 8808)
