#!/usr/bin/env python3
"""Re-export a cuDNN-based Prosit *intensity* SavedModel to a CPU+GPU-capable SavedModel.

The original Prosit 2019/2020 intensity cores are TF1 Keras graphs whose GRUs are baked as the
GPU-only `CudnnRNN` op, so they fail to load on CPU. This rebuilds the identical architecture in
TF2 Keras with `GRU(reset_after=True)` (which runs on CPU and, on GPU, uses cuDNN via the
implementation_selector dual-path), loads the original trained weights applying the documented
CuDNNGRU->GRU layout conversion (keras PR #9112), and exports a SavedModel whose serving signature
matches the original so it drops into koina with NO config/ensemble change.

Validated on Prosit_2019_intensity: end-to-end output is numerically identical to the GPU server
(cosine 1.000000, maxAbsDiff ~1e-6). The 2020 CID/HCD/TMT intensity cores share this architecture;
verify each with tools/cpu_reexport/validate_parity.py before publishing.

Usage (inside the converter image, see Dockerfile):
    python reexport_prosit_intensity.py /path/to/original/model.savedmodel /path/to/out/cpu_savedmodel
"""
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.core.protobuf import saved_model_pb2

SRC, OUT = sys.argv[1], sys.argv[2]


# --- Prosit attention layers (dlomix architecture) ---
class DecoderAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, time_steps, **kw):
        super().__init__(**kw)
        self.time_steps = time_steps

    def build(self, s):
        self.permute = tf.keras.layers.Permute((2, 1))
        self.dense = tf.keras.layers.Dense(self.time_steps, activation="softmax")
        self.multiply = tf.keras.layers.Multiply()
        super().build(s)

    def call(self, inputs):
        x = self.permute(inputs)
        x = self.dense(x)
        x = self.permute(x)
        return self.multiply([inputs, x])


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


def build_model(vocab=22, emb_dim=32, seq_len=30, units=(256, 512), frag=6):
    emb = tf.keras.layers.Embedding(vocab, emb_dim, name="sequence_embedding")
    se = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units[0], return_sequences=True)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(units[1], return_sequences=True),
        tf.keras.layers.Dropout(0.2)], name="sequence_encoder")
    att = AttentionLayer(name="encoder_att")
    me = tf.keras.Sequential([
        tf.keras.layers.Concatenate(),
        tf.keras.layers.Dense(units[1], name="meta_dense"),
        tf.keras.layers.Dropout(0.2)], name="meta_encoder")
    fus = tf.keras.Sequential([
        tf.keras.layers.Multiply(),
        tf.keras.layers.RepeatVector(seq_len - 1)], name="fusion")
    dec = tf.keras.Sequential([
        tf.keras.layers.GRU(units[1], return_sequences=True, name="decoder"),
        tf.keras.layers.Dropout(0.2),
        DecoderAttentionLayer(seq_len - 1)], name="decoder_seq")
    reg = tf.keras.Sequential([
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(frag), name="time_dense"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Flatten()], name="regressor")
    pep = tf.keras.Input((seq_len,), dtype=tf.int32)
    ce = tf.keras.Input((1,), dtype=tf.float32)
    ch = tf.keras.Input((6,), dtype=tf.float32)
    out = reg(dec(fus([att(se(emb(pep))), me([ce, ch])])))
    return tf.keras.Model([pep, ce, ch], out), dict(emb=emb, se=se, att=att, me=me, dec=dec, reg=reg)


# --- CuDNNGRU -> GRU(reset_after=True) weight conversion (keras PR #9112) ---
def gru_weights(rdr, prefix, units):
    def t(n):
        return rdr.get_tensor(f"{prefix}/{n}").astype(np.float32)
    kernel = np.hstack([k.T.reshape(k.shape, order="F") for k in np.hsplit(t("kernel"), 3)])
    rkernel = np.hstack([k.T for k in np.hsplit(t("recurrent_kernel"), 3)])
    bias = t("bias").reshape(2, 3 * units)
    return [kernel, rkernel, bias]


def main():
    rdr = tf.train.load_checkpoint(f"{SRC}/variables/variables")
    g = lambda n: rdr.get_tensor(n).astype(np.float32)
    model, L = build_model()
    model([np.zeros((1, 30), np.int32), np.zeros((1, 1), np.float32), np.zeros((1, 6), np.float32)])

    L["emb"].set_weights([g("embedding/embeddings")])
    L["se"].layers[0].forward_layer.set_weights(gru_weights(rdr, "encoder1/forward_encoder1_gru", 256))
    L["se"].layers[0].backward_layer.set_weights(gru_weights(rdr, "encoder1/backward_encoder1_gru", 256))
    L["se"].layers[2].set_weights(gru_weights(rdr, "encoder2", 512))
    L["att"].set_weights([g("encoder_att/encoder_att_W"), g("encoder_att/encoder_att_b")])
    L["me"].layers[1].set_weights([g("meta_dense/kernel"), g("meta_dense/bias")])
    L["dec"].layers[0].set_weights(gru_weights(rdr, "decoder", 512))
    L["dec"].layers[2].dense.set_weights([g("dense_1/kernel"), g("dense_1/bias")])
    L["reg"].layers[0].set_weights([g("timedense/kernel"), g("timedense/bias")])

    @tf.function(input_signature=[
        tf.TensorSpec((None, 30), tf.int32, name="peptides_in"),
        tf.TensorSpec((None, 1), tf.float32, name="collision_energy_in"),
        tf.TensorSpec((None, 6), tf.float32, name="precursor_charge_in")])
    def serve(peptides_in, collision_energy_in, precursor_charge_in):
        return {"out/Reshape:0": model([peptides_in, collision_energy_in, precursor_charge_in], training=False)}

    tf.saved_model.save(model, OUT, signatures={"serving_default": serve})

    # append ':0' to signature input keys so they match the koina ensemble's input_map
    p = f"{OUT}/saved_model.pb"
    sm = saved_model_pb2.SavedModel()
    sm.ParseFromString(open(p, "rb").read())
    sig = sm.meta_graphs[0].signature_def["serving_default"]
    for old in list(sig.inputs.keys()):
        if not old.endswith(":0"):
            sig.inputs[old + ":0"].CopyFrom(sig.inputs[old])
            del sig.inputs[old]
    open(p, "wb").write(sm.SerializeToString())
    print(f"Wrote CPU+GPU SavedModel to {OUT}")
    print("signature inputs:", list(sig.inputs.keys()), "outputs:", list(sig.outputs.keys()))


if __name__ == "__main__":
    main()
