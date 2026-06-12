#!/usr/bin/env python3
"""Re-export a cuDNN-based Prosit *iRT* SavedModel to a CPU+GPU-capable SavedModel.

Same recipe as reexport_prosit_intensity.py but for the simpler retention-time architecture
(dlomix PrositRetentionTimePredictor): embedding -> BiGRU(256) -> GRU(512) -> attention ->
Dense(512, relu) [pep_dense1] -> Dense(1) [prediction]. Sequence input only, scalar output.

Usage (inside the converter image):
    python reexport_prosit_irt.py SRC_savedmodel OUT_savedmodel VOCAB EMB_DIM INPUT_NAME
e.g. Prosit_2019_irt:           ... 22 32 sequence_integer
     Prosit_2019_irt_supplement: ... 22 32 sequence_integer
     Prosit_2020_irt_TMT:        ... 23 16 peptides_in:0
Output signature is always {INPUT_NAME -> prediction/BiasAdd:0} to match the koina ensemble.
"""
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.core.protobuf import saved_model_pb2

SRC, OUT, VOCAB, EMB, INPUT_NAME = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.init = initializers.get("glorot_uniform")

    def build(self, s):
        self.W = self.add_weight(shape=(s[-1],), initializer=self.init, name=f"{self.name}_W")
        self.b = self.add_weight(shape=(s[1],), initializer="zero", name=f"{self.name}_b")
        super().build(s)

    def call(self, x, mask=None):
        a = K.squeeze(K.dot(x, K.expand_dims(self.W)), -1) + self.b
        a = K.exp(K.tanh(a))
        a /= K.cast(K.sum(a, 1, keepdims=True) + K.epsilon(), K.floatx())
        return K.sum(x * K.expand_dims(a), 1)


def gru_weights(rdr, prefix, units):
    t = lambda n: rdr.get_tensor(f"{prefix}/{n}").astype(np.float32)
    kernel = np.hstack([k.T.reshape(k.shape, order="F") for k in np.hsplit(t("kernel"), 3)])
    rkernel = np.hstack([k.T for k in np.hsplit(t("recurrent_kernel"), 3)])
    return [kernel, rkernel, t("bias").reshape(2, 3 * units)]


def main():
    rdr = tf.train.load_checkpoint(f"{SRC}/variables/variables")
    g = lambda n: rdr.get_tensor(n).astype(np.float32)

    emb = tf.keras.layers.Embedding(VOCAB, EMB, name="sequence_embedding")
    se = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.GRU(512, return_sequences=True),
        tf.keras.layers.Dropout(0.5)], name="encoder")
    att = AttentionLayer(name="encoder_att")
    pep_dense1 = tf.keras.layers.Dense(512, activation="relu", name="pep_dense1")
    prediction = tf.keras.layers.Dense(1, name="prediction")

    pep = tf.keras.Input((30,), dtype=tf.int32)
    out = prediction(pep_dense1(att(se(emb(pep)))))
    model = tf.keras.Model(pep, out)
    model(np.zeros((1, 30), np.int32))

    emb.set_weights([g("embedding/embeddings")])
    se.layers[0].forward_layer.set_weights(gru_weights(rdr, "encoder1/forward_encoder1_gru", 256))
    se.layers[0].backward_layer.set_weights(gru_weights(rdr, "encoder1/backward_encoder1_gru", 256))
    se.layers[2].set_weights(gru_weights(rdr, "encoder2", 512))
    att.set_weights([g("encoder_att/encoder_att_W"), g("encoder_att/encoder_att_b")])
    pep_dense1.set_weights([g("pep_dense1/kernel"), g("pep_dense1/bias")])
    prediction.set_weights([g("prediction/kernel"), g("prediction/bias")])

    @tf.function(input_signature=[tf.TensorSpec((None, 30), tf.int32, name="seq")])
    def serve(seq):
        return {"prediction/BiasAdd:0": model(seq, training=False)}

    tf.saved_model.save(model, OUT, signatures={"serving_default": serve})

    # rename the single signature input key to the name the ensemble expects
    p = f"{OUT}/saved_model.pb"
    sm = saved_model_pb2.SavedModel()
    sm.ParseFromString(open(p, "rb").read())
    sig = sm.meta_graphs[0].signature_def["serving_default"]
    old = list(sig.inputs.keys())[0]
    sig.inputs[INPUT_NAME].CopyFrom(sig.inputs[old])
    del sig.inputs[old]
    open(p, "wb").write(sm.SerializeToString())
    print(f"Wrote {OUT}; inputs={list(sig.inputs.keys())} outputs={list(sig.outputs.keys())}")


if __name__ == "__main__":
    main()
