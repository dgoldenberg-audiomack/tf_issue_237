# tf_issue_237

This repository contains all the code necessary to reproduce the following issue:
https://stackoverflow.com/questions/66389284/what-does-this-error-mean-and-how-to-troubleshoot-it-valueerror-shape-must-b

Pre-requisites:
- python 3, e.g. I'm using Python 3.7.3
- Spark 3.0.1
- tensorflow==2.4.0
- tensorflow_recommenders==v0.4.0
- tensorflow-io-nightly==0.17.0.dev20210208174016
- Note on TF IO:
  - I started with tensorflow_io==0.17.0, that yielded: https://github.com/tensorflow/io/issues/1254
  - I then ported to tensorflow-io-nightly which contains the fix for that issue
  - Since then, however, I've come across this: https://github.com/tensorflow/io/issues/1313
  - Per that ticket, it was recommended to me to use tensorflow-io-nightly=<earlier than 2021/02/10>.
  - Therefore, I chose tensorflow-io-nightly==0.17.0.dev20210208174016

To execute code to reproduce the issue:
```
$ python recsys_tf_237/recsystf_driver.py
```

The code:
- recsystf_driver.py - the Spark driver to execute
- recsystf_model.py - the code exercising TF recommenders

The input data:
- input-data/events - contains Parquet with event (interactions) data (user id's with item id's)
- input-data/items - contains CSV data with item ID's
- input-data/users - contains CSV data with user ID's

The full output with the stack trace I'm seeing:
```
21/02/26 20:07:12 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).

>> Running the TFRS based recommender...
>> Loading TF datasets...
>> Loading the ITEMS dataset from ./input-data/items...
2021-02-26 20:07:13.702327: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-02-26 20:07:13.702606: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
>> ITEMS dataset: loaded.
>> Loading the EVENTS dataset from ./input-data/events...
>> Events filepaths: ['./input-data/events/events-001.snappy.parquet']
>> Loading ./input-data/events/events-001.snappy.parquet for events...
2021-02-26 20:07:13.780001: I tensorflow_io/core/kernels/cpu_check.cc:128] Your CPU supports instructions that this TensorFlow IO binary was not compiled to use: AVX2 FMA
>> EVENTS dataset: loaded
>> Loading TF datasets: done.
>> Preparing data...
>> Data preparation: done.
WARNING:tensorflow:There are non-GPU devices in `tf.distribute.Strategy`, not using nccl allreduce.
>> Number of devices: 1
>> Training the model...
2021-02-26 20:07:27.259021: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
Epoch 1/3
WARNING:tensorflow:Model was constructed with shape (None, 1000) for input KerasTensor(type_spec=TensorSpec(shape=(None, 1000), dtype=tf.string, name='string_lookup_1_input'), name='string_lookup_1_input', description="created by layer 'string_lookup_1_input'"), but it was called on an input with incompatible shape (None,).
Traceback (most recent call last):
  File "recsys_tf_237/recsystf_driver.py", line 54, in <module>
    main(sys.argv)
  File "recsys_tf_237/recsystf_driver.py", line 38, in main
    model_maker.train_and_evaluate(model, NUM_TRAIN_EPOCHS)
  File "/Users/dgoldenberg/work/temp2/tf_issue_237/recsys_tf_237/recsystf_model.py", line 179, in train_and_evaluate
    model.fit(model.cached_train_event_ds, epochs=num_epochs)
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/keras/engine/training.py", line 1100, in fit
    tmp_logs = self.train_function(iterator)
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/eager/def_function.py", line 828, in __call__
    result = self._call(*args, **kwds)
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/eager/def_function.py", line 871, in _call
    self._initialize(args, kwds, add_initializers_to=initializers)
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/eager/def_function.py", line 726, in _initialize
    *args, **kwds))
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/eager/function.py", line 2969, in _get_concrete_function_internal_garbage_collected
    graph_function, _ = self._maybe_define_function(args, kwargs)
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/eager/function.py", line 3361, in _maybe_define_function
    graph_function = self._create_graph_function(args, kwargs)
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/eager/function.py", line 3206, in _create_graph_function
    capture_by_value=self._capture_by_value),
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/framework/func_graph.py", line 990, in func_graph_from_py_func
    func_outputs = python_func(*func_args, **func_kwargs)
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/eager/def_function.py", line 634, in wrapped_fn
    out = weak_wrapped_fn().__wrapped__(*args, **kwds)
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/framework/func_graph.py", line 977, in wrapper
    raise e.ag_error_metadata.to_exception(e)
ValueError: in user code:

    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/keras/engine/training.py:805 train_function  *
        return step_function(self, iterator)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow_recommenders/tasks/retrieval.py:157 call  *
        update_op = self._factorized_metrics.update_state(query_embeddings,
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow_recommenders/metrics/factorized_top_k.py:83 update_state  *
        top_k_predictions, _ = self._candidates(query_embeddings, k=self._k)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow_recommenders/layers/factorized_top_k.py:224 top_k  *
        joined_scores = tf.concat([state_scores, x_scores], axis=1)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/util/dispatch.py:201 wrapper  **
        return target(*args, **kwargs)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/ops/array_ops.py:1677 concat
        return gen_array_ops.concat_v2(values=values, axis=axis, name=name)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/ops/gen_array_ops.py:1208 concat_v2
        "ConcatV2", values=values, axis=axis, name=name)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py:750 _apply_op_helper
        attrs=attr_protos, op_def=op_def)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/framework/func_graph.py:592 _create_op_internal
        compute_device)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/framework/ops.py:3536 _create_op_internal
        op_def=op_def)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/framework/ops.py:2016 __init__
        control_input_ops, op_def)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/framework/ops.py:1856 _create_c_op
        raise ValueError(str(e))

    ValueError: Shape must be rank 2 but is rank 3 for '{{node concat}} = ConcatV2[N=2, T=DT_FLOAT, Tidx=DT_INT32](args_0, args_2, concat/axis)' with input shapes: [?,0], [?,?,?], [].
```


Without using the `tf.data.experimental.AutoShardPolicy.OFF`, the error is similar but https://stackoverflow.com/questions/65322700/tensorflow-keras-consider-either-turning-off-auto-sharding-or-switching-the-a recommends disabling auto-sharding. 

```
21/02/27 01:24:04 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
>> Loading TF datasets...
>> Loading the ITEMS dataset from ./input-data/items...
2021-02-27 01:24:06.223795: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-02-27 01:24:06.224062: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
>> ITEMS dataset: loaded.
>> Loading the EVENTS dataset from ./input-data/events...
>> Events filepaths: ['./input-data/events/events.snappy.parquet']
>> Loading ./input-data/events/events.snappy.parquet for events...
2021-02-27 01:24:06.287886: I tensorflow_io/core/kernels/cpu_check.cc:128] Your CPU supports instructions that this TensorFlow IO binary was not compiled to use: AVX2 FMA
>> EVENTS dataset: loaded
>> Loading TF datasets: done.
>> Preparing data...
>> Data preparation: done.
WARNING:tensorflow:There are non-GPU devices in `tf.distribute.Strategy`, not using nccl allreduce.
>> Training the model...
2021-02-27 01:24:19.974113: W tensorflow/core/grappler/optimizers/data/auto_shard.cc:656] In AUTO-mode, and switching to DATA-based sharding, instead of FILE-based sharding as we cannot find appropriate reader dataset op(s) to shard. Error: Did not find a shardable source, walked to a node which is not a dataset: name: "UnbatchDataset/_17"
op: "UnbatchDataset"
input: "MapDataset/_16"
attr {
  key: "output_shapes"
  value {
    list {
      shape {
      }
    }
  }
}
attr {
  key: "output_types"
  value {
    list {
      type: DT_STRING
    }
  }
}
. Consider either turning off auto-sharding or switching the auto_shard_policy to DATA to shard this dataset. You can do this by creating a new `tf.data.Options()` object then setting `options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA` before applying the options object to the dataset via `dataset.with_options(options)`.
2021-02-27 01:24:19.995799: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
Epoch 1/3
WARNING:tensorflow:Model was constructed with shape (None, 1000) for input KerasTensor(type_spec=TensorSpec(shape=(None, 1000), dtype=tf.string, name='string_lookup_1_input'), name='string_lookup_1_input', description="created by layer 'string_lookup_1_input'"), but it was called on an input with incompatible shape (None,).
Traceback (most recent call last):
  File "recsys_tf_237/recsystf_driver.py", line 37, in <module>
    main(sys.argv)
  File "recsys_tf_237/recsystf_driver.py", line 29, in main
    model_maker.train_and_evaluate(model, NUM_TRAIN_EPOCHS)
  File "/Users/dgoldenberg/work/audiomack-code/code/tf_issue_237/recsys_tf_237/recsystf_model.py", line 111, in train_and_evaluate
    model.fit(model.cached_train_event_ds, epochs=num_epochs)
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/keras/engine/training.py", line 1100, in fit
    tmp_logs = self.train_function(iterator)
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/eager/def_function.py", line 828, in __call__
    result = self._call(*args, **kwds)
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/eager/def_function.py", line 871, in _call
    self._initialize(args, kwds, add_initializers_to=initializers)
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/eager/def_function.py", line 726, in _initialize
    *args, **kwds))
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/eager/function.py", line 2969, in _get_concrete_function_internal_garbage_collected
    graph_function, _ = self._maybe_define_function(args, kwargs)
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/eager/function.py", line 3361, in _maybe_define_function
    graph_function = self._create_graph_function(args, kwargs)
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/eager/function.py", line 3206, in _create_graph_function
    capture_by_value=self._capture_by_value),
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/framework/func_graph.py", line 990, in func_graph_from_py_func
    func_outputs = python_func(*func_args, **func_kwargs)
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/eager/def_function.py", line 634, in wrapped_fn
    out = weak_wrapped_fn().__wrapped__(*args, **kwds)
  File "/Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/framework/func_graph.py", line 977, in wrapper
    raise e.ag_error_metadata.to_exception(e)
ValueError: in user code:

    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/keras/engine/training.py:805 train_function  *
        return step_function(self, iterator)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow_recommenders/tasks/retrieval.py:157 call  *
        update_op = self._factorized_metrics.update_state(query_embeddings,
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow_recommenders/metrics/factorized_top_k.py:83 update_state  *
        top_k_predictions, _ = self._candidates(query_embeddings, k=self._k)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow_recommenders/layers/factorized_top_k.py:224 top_k  *
        joined_scores = tf.concat([state_scores, x_scores], axis=1)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/util/dispatch.py:201 wrapper  **
        return target(*args, **kwargs)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/ops/array_ops.py:1677 concat
        return gen_array_ops.concat_v2(values=values, axis=axis, name=name)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/ops/gen_array_ops.py:1208 concat_v2
        "ConcatV2", values=values, axis=axis, name=name)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py:750 _apply_op_helper
        attrs=attr_protos, op_def=op_def)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/framework/func_graph.py:592 _create_op_internal
        compute_device)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/framework/ops.py:3536 _create_op_internal
        op_def=op_def)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/framework/ops.py:2016 __init__
        control_input_ops, op_def)
    /Users/dgoldenberg/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow/python/framework/ops.py:1856 _create_c_op
        raise ValueError(str(e))

    ValueError: Shape must be rank 2 but is rank 3 for '{{node concat}} = ConcatV2[N=2, T=DT_FLOAT, Tidx=DT_INT32](args_0, args_2, concat/axis)' with input shapes: [?,0], [?,?,?], [].
```

