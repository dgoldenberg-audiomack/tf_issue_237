import sys
import time

import tensorflow as tf
from pyspark.sql import SparkSession

from recsys_tf_237.recsystf_model import TfrsModelMaker

NUM_TRAIN_EPOCHS = 3

items_path = "./input-data/items"
users_path = "./input-data/users"
events_path = "./input-data/events"
num_items = 494433
num_users = 3630815
num_events = 10183074


def main(args):
    # Using allow_soft_placement=True allows TF to fall back to CPU when no GPU implementation is available.
    tf.config.set_soft_device_placement(True)

    start_time = time.time()
    spark = SparkSession.builder.appName("Recsys-TFRS").getOrCreate()

    print()
    print(">> Running the TFRS based recommender...")

    # Load the intermediary parquet data into TF datasets
    model_maker = TfrsModelMaker(items_path, users_path, events_path, num_items, num_users, num_events)
    # Build the model
    model = model_maker.create_model()
    # Train and evaluate the model
    model_maker.train_and_evaluate(model, NUM_TRAIN_EPOCHS)

    # Get recomms
    model_maker.generate_recommendations(model, model_maker.items_ds)

    elapsed_time = time.time() - start_time
    str_elapsed_time = time.strftime("%H : %M : %S", time.gmtime(elapsed_time))

    print()
    print(">> Done. Duration: {}.".format(str_elapsed_time))
    print()

    spark.stop()


if __name__ == "__main__":
    main(sys.argv)
