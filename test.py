import tensorflow as tf

max_sentence_len = 80

def read_csv(batch_size, file_name):
    filename_queue = tf.train.string_input_producer([file_name])
    reader = tf.TextLineReader(skip_header_lines=0)
    key, value = reader.read(filename_queue)
    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(
        value,
        field_delim=' ',
        record_defaults=[[0] for i in range(80 * 2)])

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)


whole = read_csv(2, "train.txt")
features = tf.transpose(tf.stack(whole[0:max_sentence_len]))
label = tf.transpose(tf.stack(whole[max_sentence_len:]))

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1):
    # Retrieve a single instance:
    x, y, z = sess.run([features, label, whole])
    print(len(z))
    print(z[0])
    print("*" * 100)
    print(x)

  coord.request_stop()
  coord.join(threads)


