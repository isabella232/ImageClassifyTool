import itertools
import threading
import tensorflow as tf
import numpy as np
from threading import Thread, Event

size = 10240
def myfunc(sess, name):
  values = set()
  count = 0
  while True:
    count += 1
    v = sess.run(name + "/matmul4:0", feed_dict={name + "/input:0": np.ones((1,1))})
    v = float(np.squeeze(v))
    old = len(values)
    values.add(v)
    if len(values) != old:
      print(values, name, count)

def create_graph(sess, name):
  with sess.graph.as_default():
    with tf.variable_scope(name):
      input = tf.placeholder(tf.float32, shape=[1,1], name = "input")
      tf.set_random_seed(1)

      matrix1 = tf.Variable(tf.truncated_normal([1, size]), name = 'matrix1')
      matrix2 = tf.Variable(tf.truncated_normal([size, size]), name = 'matrix2')
      matrix4 = tf.Variable(tf.truncated_normal([size, 1]), name = 'matrix4')

      matmul1 = tf.matmul(input, matrix1, name = 'matmul1')
      matmul2 = tf.matmul(matmul1, matrix2, name = 'matmul2')
      matmul4 = tf.matmul(matmul2, matrix4, name = "matmul4")
      sess.run(tf.global_variables_initializer())
      print("finish - ", name)

sess1 = tf.Session()
with tf.device("/gpu:0"):
  create_graph(sess1, "s1")

sess2 = tf.Session()
with tf.device("/gpu:1"):
  create_graph(sess2, "s2")

def main():
    tlist =[]
    t1 = Thread(target=myfunc, args=(sess1, 's1'))
    tlist.append(t1)
    #t1.start()

    t2 = Thread(target=myfunc, args=(sess2, 's2'))
    tlist.append(t2)
    #t2.start()
    for thread in tlist:
        thread.start()
    for thread in tlist:
        thread.join()

    print("done - all")

if __name__== "__main__":
    main()

'''
  import tensorflow as tf

visible_device_list = '1' # use , separator for more GPUs like '0, 1'
per_process_gpu_memory_fraction = 0.9 # avoid out of memory
intra_op_parallelism_threads = 2  # default in tensorflow
inter_op_parallelism_threads = 5  # default in tensorflow

gpu_options = tf.compat.v1.GPUOptions(allow_growth = True)


config = tf.compat.v1.ConfigProto(
         allow_soft_placement = True,
         log_device_placement = False,
         intra_op_parallelism_threads = intra_op_parallelism_threads,
         inter_op_parallelism_threads = inter_op_parallelism_threads,
         gpu_options = gpu_options)

config = tf.compat.v1.ConfigProto(
         gpu_options = gpu_options)

s = config.SerializeToString()
# print(list(map(hex, s)))  # print by json if need

print('a serialized protobuf string for TF_SetConfig, note the byte order is in normal order.')
b = ''.join(format(b,'02x') for b in s)
print('0x%s' % b) # print by hex format

'''