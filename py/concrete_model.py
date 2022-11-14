import os

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

if __name__ == "__main__":
    model_path = os.path.join(os.getcwd(), "py\\logs\\saved_model\\model.h5")
    model = tf.keras.models.load_model(model_path)

    x = tf.TensorSpec(model.input_shape, tf.float32, name="x")
    concrete_function = tf.function(lambda x: model(x)).get_concrete_function(x)

    # now all variables are converted to constants.
    # if this step is omitted, dumped graph does not include trained weights
    frozen_model = convert_variables_to_constants_v2(concrete_function)
    print(f"{frozen_model.inputs=}")
    print(f"{frozen_model.outputs=}")

    directory = "model/frozen_model"
    tf.io.write_graph(frozen_model.graph, directory, "dqn.pb", as_text=False)
