# TensorFlowImageClassification
Scripts necesarios para entrenar un modelo que clasifique imágenes y el resultado pueda convertirse a un modelo en CoreML

## Intro

Hacer intro...

## Scripts interesantes


## Jupyter


## ¿Cómo probar el modelo?



## ¿Cómo obtengo info del modelo generado?
El archivo inspect_pb.py permite inspeccionar el grafo generado y determinar, entro otras cosas el atributo **output_feature_names** necesario para realizar la conversión a CoreML.
Para ejecutarlo:

```

python3 inspect_pb.py path_al_archivo_pb pat_al_archivo_salida.txt

```

Por ejemplo:

```

python3 inspect_pb.py tf_files/retrained_graph.pb info.txt

```

Luego en el arhivo txt generado se debe buscar soft_max:

```
559: op name = import/final_result, op type = ( Softmax ), inputs = import/final_training_ops/Wx_plus_b/add:0, outputs = import/final_result:0
@input shapes:
name = import/final_training_ops/Wx_plus_b/add:0 : (?, 2)
@output shapes:
name = *import/final_result:0* : (?, 2)

```
y el **name** del **@output shapes:** debe utilizarse en el script python conversor convertToCoreML.py:


```
import tfcoreml as tf_converter

tf_converter.convert(tf_model_path = 'tf_files/retrained_graph.pb',
                     mlmodel_path = 'maradona.mlmodel',
                     image_input_names = 'input:0',
                     class_labels = 'tf_files/retrained_labels.txt',
                     output_feature_names = ['final_result:0'],
                     red_bias = -1,
                     green_bias = -1,
                     blue_bias = -1,
                     image_scale = 2.0/255.0)	
```






