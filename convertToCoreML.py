import tfcoreml as tf_converter

class_labels = 'imagenet_slim_labels.txt'

tf_converter.convert(tf_model_path = 'tf_files/retrained_graph.pb',
                     mlmodel_path = 'maradona.mlmodel',
                     image_input_names = 'input:0',
                     class_labels = 'tf_files/retrained_labels.txt',
                     output_feature_names = ['final_result:0'],
                     red_bias = -1,
                     green_bias = -1,
                     blue_bias = -1,
                     image_scale = 2.0/255.0)					
