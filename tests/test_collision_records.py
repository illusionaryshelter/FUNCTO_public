"""Exact bounded array records and single-state collision-selection archives."""

# BEGIN malformed records cannot allocate an unbounded collision inventory.
import base64
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from manipulation.collision_records import MAX_ARRAY_BYTES, decode_array, encode_array, slice_layer_selection
from manipulation.voxel_layers import LayerSelection


def selection_record():
    shape = (2, 3, 2)
    fine = torch.arange(12).reshape(shape).remainder(2).bool()
    envelope = torch.arange(72, dtype=torch.float64).reshape(*shape, 2, 3)
    lower = torch.arange(36, dtype=torch.int64).reshape(*shape, 3)
    selected = LayerSelection(fine, fine, envelope, lower, lower+3, ("arm", "tool"))
    return selected.record()


class CollisionRecordTests(unittest.TestCase):
    def test_numpy_and_detached_cpu_tensor_roundtrip_exact_stored_bytes(self):
        arrays = (np.array([[True, False], [False, True]]),
                  np.array([-2**63, 0, 2**63-1], dtype=np.int64),
                  np.array([-0., .1, np.nextafter(np.float32(1.), np.float32(2.))], dtype=np.float32),
                  np.array([-0., .1, np.nextafter(1., 2.)], dtype=np.float64))
        for array in arrays:
            for supplied in (array, torch.from_numpy(array)):
                with self.subTest(dtype=array.dtype, source=type(supplied).__name__):
                    record = encode_array(supplied)
                    actual = decode_array(record)
                    self.assertEqual(actual.dtype, array.dtype)
                    self.assertEqual(actual.shape, array.shape)
                    self.assertEqual(actual.tobytes(), array.tobytes())
                    self.assertFalse(actual.flags.writeable)
                    self.assertEqual(len(record["data_base64"]), 4*((array.nbytes+2)//3))

    def test_big_endian_and_noncontiguous_inputs_write_canonical_little_endian(self):
        supplied = np.arange(24, dtype=">f8").reshape(4, 6)[:, ::2]
        self.assertFalse(supplied.flags.c_contiguous)
        record = encode_array(supplied)
        self.assertEqual(record["dtype"], "<f8")
        np.testing.assert_array_equal(decode_array(record), supplied)

    def test_scalar_and_zero_size_arrays_keep_their_shapes(self):
        for supplied in (np.array(4., dtype=np.float32), np.empty((0, 3), dtype=np.int64)):
            record = encode_array(supplied)
            self.assertEqual(decode_array(record).shape, supplied.shape)
            np.testing.assert_array_equal(decode_array(record), supplied)

    def test_encoder_rejects_untyped_lists_graph_tensors_and_unsupported_dtypes(self):
        for supplied in ([1., 2.], torch.ones(2, requires_grad=True), np.ones(2, dtype=object),
                         np.ones(2, dtype=np.complex64), np.ones(2, dtype=np.float16)):
            with self.subTest(source=type(supplied).__name__), self.assertRaises(ValueError):
                encode_array(supplied)

    def test_oversized_encoding_is_rejected_before_payload_copy(self):
        # A broadcast view proves admission without actually allocating 32 MiB.
        oversized = np.broadcast_to(np.array(1., dtype=np.float32), (MAX_ARRAY_BYTES//4+1,))
        with patch("manipulation.collision_records.base64.b64encode") as encoder:
            with self.assertRaisesRegex(ValueError, "32 MiB"):
                encode_array(oversized)
            encoder.assert_not_called()

    def test_bad_shape_dtype_size_and_keys_fail_before_base64_decoding(self):
        valid = encode_array(np.ones(2, dtype=np.float64))
        malformed = ([1., 2.], {**valid, "extra": True}, {"shape": [2], "dtype": "<f8"},
                     {**valid, "shape": [True]}, {**valid, "shape": [1.5]},
                     {**valid, "shape": [-1]}, {**valid, "shape": [1]*6},
                     {**valid, "shape": [2**32, 2**32, 3]},
                     {**valid, "shape": [2**63, 0]},
                     {**valid, "shape": [MAX_ARRAY_BYTES//8+1]},
                     {**valid, "dtype": ">f8"}, {**valid, "dtype": "float64"},
                     {**valid, "dtype": "|O"}, {**valid, "data_base64": b"AAAA"})
        for record in malformed:
            with self.subTest(record=record), patch("manipulation.collision_records.base64.b64decode") as decoder:
                with self.assertRaises(ValueError):
                    decode_array(record)
                decoder.assert_not_called()

    def test_invalid_base64_and_decoded_byte_count_are_rejected(self):
        record = encode_array(np.array([1.], dtype=np.float32))
        for payload in ("!!!!!!!!", "éAAAAAAA", "AAAAAAA=", "AAAA", "AAAAAA==\n"):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                decode_array({**record, "data_base64": payload})
        # Encoded length alone cannot distinguish one byte from three bytes.
        with self.assertRaisesRegex(ValueError, "Decoded.*byte length"):
            decode_array({"shape": [1], "dtype": "|b1", "data_base64": base64.b64encode(b"abc").decode()})

    def test_layer_selection_record_and_state_slice_keep_typed_values(self):
        original = selection_record()
        records = {"robot/background": original, "robot/owner": deepcopy(original)}
        sliced = slice_layer_selection(records, index=(1, 2))
        for target in records:
            self.assertEqual(sliced[target]["body_ids"], ["arm", "tool"])
            self.assertIsNot(sliced[target]["body_ids"], original["body_ids"])
            for name in ("body_fine", "influence_aabb_grid_m", "index_lower", "index_upper"):
                actual, full = decode_array(sliced[target][name]), decode_array(original[name])
                self.assertEqual(actual.shape[:2], (1, 1))
                self.assertEqual(actual.dtype, full.dtype)
                np.testing.assert_array_equal(actual, full[1:2, 2:3])
        self.assertEqual(decode_array(records["robot/background"]["body_fine"]).shape, (2, 3, 2))

    def test_state_slice_rejects_incomplete_misaligned_or_out_of_range_records(self):
        original = selection_record()
        malformed = []
        for name, value in (("body_ids", ["arm", "arm"]), ("body_ids", ["arm"]),
                            ("body_fine", encode_array(np.zeros((2, 3, 2), dtype=np.float64))),
                            ("influence_aabb_grid_m", encode_array(np.zeros((2, 2, 2, 2, 3)))),
                            ("index_lower", encode_array(np.zeros((2, 3, 2, 3), dtype=np.float64)))):
            malformed.append({**original, name: value})
        malformed.append({name: value for name, value in original.items() if name != "index_upper"})
        for record in malformed:
            with self.subTest(keys=list(record)), self.assertRaises(ValueError):
                slice_layer_selection({"robot/background": record})
        for index in ((2, 0), (0, 3), (-1, 0), (True, 0), (0,), [0, 0]):
            with self.subTest(index=index), self.assertRaises(ValueError):
                slice_layer_selection({"robot/background": original}, index=index)
        self.assertEqual(slice_layer_selection({}), {})


if __name__ == "__main__":
    unittest.main()
# END malformed, oversized or ambiguous records cannot enter independent checks.
