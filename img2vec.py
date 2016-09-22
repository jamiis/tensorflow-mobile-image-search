# python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# retrain.py downloaded with setup.sh
import retrain
POOL3 = retrain.BOTTLENECK_TENSOR_NAME

# tensorflow
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

# other
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Output feature vector '
                                    'of image according to inception v3.')
    parser.add_argument('-i','--image_path',
                        required=True,
                        help='Path to image to be transformed into a feature vector.') 
    parser.add_argument('-o','--output_graph',
                        default='/tmp/output_graph.pb',
                        help='Where to save the trained graph.')
    parser.add_argument('-s','--summaries_dir', 
                        default='/tmp/img2vec_logs',
                        help='Where to save summary logs for TensorBoard.')
    parser.add_argument('-m','--model_dir', 
                        default='/tmp/imagenet',
                        help='Path to classify_image_graph_def.pb, '
                        'imagenet_synset_to_human_label_map.txt, and '
                        'imagenet_2012_challenge_label_map_proto.pbtxt.')
    return parser.parse_args()



def main(argv=None):
    if argv:
        arg = parse_args()

    # get image data
    if not gfile.Exists(arg.image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = gfile.FastGFile(arg.image_path, 'rb').read()

    # create graph which includes the penultimate tensor, pool3
    graph, pool3_tensor, jpeg_data_tensor, resized_image_tensor = (
            retrain.create_inception_graph())

    sess = tf.Session()

    # pass image through network up until just before softmax
    pool3_values = retrain.run_bottleneck_on_image(sess, image_data,
                                                   jpeg_data_tensor,
                                                   pool3_tensor)
    print(pool3_values)

    # Write out the trained graph with the weights stored as constants.
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [POOL3])
    with gfile.FastGFile(arg.output_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
        print('graph written to ', arg.output_graph)

if __name__ == '__main__':
    tf.app.run()
