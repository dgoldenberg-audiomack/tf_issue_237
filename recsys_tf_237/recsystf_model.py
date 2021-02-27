import os
import pprint
from abc import ABC
from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_recommenders as tfrs
import urllib3


class TfrsModel(tfrs.Model, ABC):
    def __init__(self, user_model, item_model, loss_task, cached_train_event_ds, cached_test_event_ds):
        super().__init__()
        self.item_model: tf.keras.Model = item_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = loss_task

        self.cached_train_event_ds = cached_train_event_ds
        self.cached_test_event_ds = cached_test_event_ds

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # Pick out the item features and pass them into the item model, getting embeddings back.
        item_embeddings = self.item_model(features["item_id"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, item_embeddings)


class TfrsModelMaker(object):
    def __init__(self, items_path, users_path, events_path, num_items, num_users, num_events):
        self.items_path = items_path
        self.users_path = users_path
        self.events_path = events_path
        self.num_items = num_items
        self.num_users = num_users
        self.num_events = num_events

        # Turn off the many Unverified HTTPS request warnings during file downloads.
        urllib3.disable_warnings()
        self.items_ds, self.events_ds = self._load_tf_datasets()
        self.test_events_ds, self.train_events_ds = self._prepare_data()


    def create_model(self):
        embedding_dimension = 32

        # Define the distribution strategy
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            tf.config.set_soft_device_placement(True)

            user_ids_filepath = self.get_filepaths(self.users_path, "*.csv")[0]
            item_ids_filepath = self.get_filepaths(self.items_path, "*.csv")[0]

            # The query tower
            u_lookup = tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=user_ids_filepath, mask_token=None
            )
            user_model = tf.keras.Sequential(
                [
                    u_lookup,
                    # We add an additional embedding to account for unknown tokens.
                    tf.keras.layers.Embedding(u_lookup.vocab_size() + 1, embedding_dimension),
                ]
            )

            # The candidate tower
            c_lookup = tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=item_ids_filepath, mask_token=None
            )
            item_model = tf.keras.Sequential(
                [
                    c_lookup,
                    # We add an additional embedding to account for unknown tokens.
                    tf.keras.layers.Embedding(c_lookup.vocab_size() + 1, embedding_dimension),
                ]
            )

            # Metrics
            cands = self.items_ds.batch(128).map(item_model)
            metrics = tfrs.metrics.FactorizedTopK(candidates=cands)

            # Loss
            task = tfrs.tasks.Retrieval(metrics=metrics)

            cached_train_event_ds = self.train_events_ds.batch(409600).cache()  # 204800
            cached_test_event_ds = self.test_events_ds.batch(204800).cache()  # 102400

            # Prevent an auto-sharding related error.
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            cached_train_event_ds = cached_train_event_ds.with_options(options)
            cached_test_event_ds = cached_test_event_ds.with_options(options)

            model = TfrsModel(user_model, item_model, task, cached_train_event_ds, cached_test_event_ds)

            model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

        return model

    @staticmethod
    def train_and_evaluate(model, num_epochs):
        print(">> Training the model...")

        # Train the model
        model.fit(model.cached_train_event_ds, epochs=num_epochs)
        # model.fit(model.cached_train_event_ds, epochs=num_epochs, verbose=0)
        print(">> Training of the model: done.")

        # Evaluate the model...

    def _load_tf_datasets(self):
        print(">> Loading TF datasets...")

        print(">> Loading the ITEMS dataset from {}...".format(self.items_path))
        file_list = self.get_filepaths(self.items_path, "*.csv")
        items_ds = tf.data.experimental.make_csv_dataset(
            file_list, column_names=["item_id"], batch_size=1000, num_parallel_reads=50, sloppy=True,
        )
        items_ds = items_ds.map(lambda x: x["item_id"])
        print(">> ITEMS dataset: loaded.")

        # dataset_peek("items_ds", items_ds)

        print(">> Loading the EVENTS dataset from {}...".format(self.events_path))
        # Load the events
        events_filepaths = self.get_filepaths(self.events_path, "*.parquet")
        print(">> Events filepaths: " + str(events_filepaths))
        events_columns = ["user_id", "item_id"]
        events_ds = self.load_dataset("events", events_filepaths, events_columns)
        events_ds = events_ds.map(lambda x: {"item_id": x["item_id"], "user_id": x["user_id"]})
        print(">> EVENTS dataset: loaded")

        # dataset_peek("events_ds", events_ds)

        print(">> Loading TF datasets: done.")

        return items_ds, events_ds

    @staticmethod
    def load_dataset(ds_name, files, columns):
        print(f">> Loading {files[0]} for {ds_name}...")
        dataset = tfio.IODataset.from_parquet(files[0], columns=columns)

        for file_name in files[1:]:
            print(f">> Loading {file_name} for {ds_name}...")
            ds = tfio.IODataset.from_parquet(file_name, columns=columns)
            dataset = dataset.concatenate(ds)

        return dataset

    @staticmethod
    def get_filepaths(dirpath, pattern):
        return tf.io.gfile.glob(os.path.join(dirpath, pattern))

    def _prepare_data(self):
        print(">> Preparing data...")

        size_80_percent = int(self.num_events * 0.8)
        size_20_percent = self.num_events - size_80_percent

        # shuffled_events_ds = self.events_ds.shuffle(self.num_events, seed=42, reshuffle_each_iteration=False)

        # These are already shuffled.
        train_events_ds = self.events_ds.take(size_80_percent)
        test_events_ds = self.events_ds.skip(size_80_percent).take(size_20_percent)

        print(">> Data preparation: done.")

        return test_events_ds, train_events_ds

