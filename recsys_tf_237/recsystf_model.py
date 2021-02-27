import os
import pprint
from abc import ABC
from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_recommenders as tfrs
import urllib3


def dataset_peek(name, ds):
    print()
    print("*" * 80)
    print("@@@ dataset: {}".format(name))
    print(ds)
    element = ds.take(1)
    print("@@@ record type: {}".format(type(element)))
    for x in element.as_numpy_iterator():
        print("@@@ x type: {}".format(type(x)))
        if isinstance(x, list):
            print("@@@ x is a list")
            pprint.pprint(x[0])
        elif isinstance(x, tuple):
            print("@@@ x is a tuple")
            pprint.pprint(x)
        elif isinstance(x, np.ndarray):
            print("@@@ x is ndarray")
            pprint.pprint(x[0])
        else:
            pprint.pprint(x)
        break
    print("*" * 80)
    print()


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

        # dataset_peek("items_ds in init", self.items_ds)
        # dataset_peek("events_ds in init", self.events_ds)

        self.test_events_ds, self.train_events_ds = self._prepare_data()
        # dataset_peek("train_events_ds in init", self.train_events_ds)
        # dataset_peek("test_events_ds in init", self.test_events_ds)

    def create_model(self):
        embedding_dimension = 32

        # Define the distribution strategy
        strategy = tf.distribute.MirroredStrategy()

        print(">> Number of devices: {}".format(strategy.num_replicas_in_sync))

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
            # print()
            # print("*" * 80)
            # print(">> candidates:")
            # print(cands)
            # print("*" * 80)
            # print()
            metrics = tfrs.metrics.FactorizedTopK(candidates=cands)
            # print()
            # print("*" * 80)
            # print(">> metrics:")
            # print(metrics)
            # print("*" * 80)
            # print()

            # Loss
            task = tfrs.tasks.Retrieval(metrics=metrics)

            # cached_train_event_ds = self.train_events_ds.batch(8192).cache()
            # cached_test_event_ds = self.test_events_ds.batch(4096).cache()

            cached_train_event_ds = self.train_events_ds.batch(409600).cache()  # 204800
            cached_test_event_ds = self.test_events_ds.batch(204800).cache()  # 102400

            # dataset_peek("cached_train_event_ds in create_model", cached_train_event_ds)
            # dataset_peek("cached_test_event_ds in create_model", cached_test_event_ds)

            # Prevent an auto-sharding related error.
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            cached_train_event_ds = cached_train_event_ds.with_options(options)
            cached_test_event_ds = cached_test_event_ds.with_options(options)

            # dataset_peek("cached_train_event_ds in create_model - 2", cached_train_event_ds)
            # dataset_peek("cached_test_event_ds in create_model - 2", cached_test_event_ds)

            model = TfrsModel(user_model, item_model, task, cached_train_event_ds, cached_test_event_ds)

            model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
            # model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1), run_eagerly=True)
            # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))

        return model

    @staticmethod
    def train_and_evaluate(model, num_epochs):
        print(">> Training the model...")

        # print()
        # print("*" * 80)
        # print(">> cached_train_event_ds:")
        # print("*" * 80)
        # i = 1
        # for x in model.cached_train_event_ds:
        #     iid = x["item_id"]
        #     uid = x["user_id"]
        #     print(">> i: {}, x size: {}, iid size: {}, uid size: {}".format(i, len(x), len(iid), len(uid)))
        #     i = i + 1
        # print("*" * 80)
        # print()

        # Train the model
        model.fit(model.cached_train_event_ds, epochs=num_epochs)
        # model.fit(model.cached_train_event_ds, epochs=num_epochs, verbose=0)
        print(">> Training of the model: done.")

        # Evaluate the model
        print(">> Evaluating the model...")
        eval_results = model.evaluate(model.cached_test_event_ds, return_dict=True)
        print(">> Evaluation of the model: done.")

        print()
        print(f">> Eval results (epochs={num_epochs}):")
        print(str(eval_results))
        print()

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

    @staticmethod
    def generate_recommendations(model, items_ds):
        # Making predictions

        # Create a model that takes in raw query features, and
        index = tfrs.layers.factorized_top_k.BruteForce(model.user_model, k=20)
        # recommends items out of the entire items dataset.
        index.index(items_ds.batch(100).map(model.item_model), items_ds)

        # Extract recs for input users ...
