import sys

import tensorflow as tf
from pyspark.sql import SparkSession

from recsys_tf_237.recsystf_model import TfrsModelMaker

NUM_TRAIN_EPOCHS = 3

items_path = "./input-data/items"
users_path = "./input-data/users"
events_path = "./input-data/events"
num_items = 494433
num_users = 3827078
num_events = 6757870


def main(args):
    # Using allow_soft_placement=True allows TF to fall back to CPU when no GPU implementation is available.
    tf.config.set_soft_device_placement(True)

    spark = SparkSession.builder.appName("Recsys-TFRS").getOrCreate()

    # Load the intermediary parquet data into TF datasets
    model_maker = TfrsModelMaker(items_path, users_path, events_path, num_items, num_users, num_events)
    # Build the model
    model = model_maker.create_model()
    # Train and evaluate the model
    model_maker.train_and_evaluate(model, NUM_TRAIN_EPOCHS)

    # Generate recomms...

    spark.stop()


if __name__ == "__main__":
    main(sys.argv)
