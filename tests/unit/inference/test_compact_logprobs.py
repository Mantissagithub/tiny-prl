import numpy as np
import pybase64
from vllm.logprobs import FlatLogprobs, Logprob

from prime_rl.inference.vllm.compact_logprobs import serialize_compact_logprobs


def test_compact_logprobs_keeps_exact_finite_support_rows():
    logprobs = FlatLogprobs()
    logprobs.append({5: Logprob(-0.2, 0, None), 7: Logprob(float("-inf"), 1, None)})
    logprobs.append({3: Logprob(-0.4, 2, None)})

    packed = serialize_compact_logprobs(logprobs, num_top_logprobs=2, drop_filtered=True)
    data = pybase64.b64decode(packed["data"])
    ids = np.frombuffer(data[:8], dtype="<u4")
    values = np.frombuffer(data[8:], dtype="<f4")

    assert packed["format"] == "u32-f32-le-v1"
    assert packed["offsets"] == [0, 1, 2]
    np.testing.assert_array_equal(ids, [5, 3])
    np.testing.assert_allclose(values, [-0.2, -0.4])
