import tensorflow as tf
from src import config

def postprocess(probs):
    max_values = tf.reduce_max(probs, axis=-1, keepdims=True)
    mask = tf.equal(probs, max_values)
    indices = tf.where(mask)
    argmax_indices = indices[:, -1]
    argmax_plus_one = argmax_indices + 1
    return tf.cast(argmax_plus_one, dtype=probs.dtype)


input_spec = {'review_text': tf.TensorSpec((None, ), tf.string, 'review_text')}


def export_model(model, path):
    @tf.function(input_signature=[input_spec])
    def serving_fn(input):
        probs = model(input['review_text'])
        label = postprocess(probs)
        return {'rating_pred': label}

    tf.saved_model.save(
        model,
        export_dir=path,
        signatures={config.SIG_NAME: serving_fn}
    )