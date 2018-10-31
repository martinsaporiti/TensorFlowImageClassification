IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"

python3 retrain.py \
   --bottleneck_dir=tf_files/bottlenecks --how_many_training_steps=1000 \
   --model_dir=tf_files/models/  --summaries_dir=tf_files/training_summaries/ \
   --output_graph=tf_files/retrained_graph.pb   --output_labels=tf_files/retrained_labels.txt --architecture="${ARCHITECTURE}"  \
   --image_dir=data
