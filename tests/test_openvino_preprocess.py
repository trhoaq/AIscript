import unittest

import numpy as np

IMPORT_ERROR = None
try:
    from core.openvino_preprocess import IMAGENET_MEAN, IMAGENET_STD, preprocess_bgr_frame
except ModuleNotFoundError as exc:
    IMPORT_ERROR = exc
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    preprocess_bgr_frame = None  # type: ignore[assignment]


@unittest.skipIf(IMPORT_ERROR is not None, "opencv-python (cv2) is not installed in the test environment.")
class OpenVinoPreprocessTests(unittest.TestCase):
    def test_preprocess_returns_nchw_tensor(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        blob = preprocess_bgr_frame(image, img_size=256)

        self.assertIsInstance(blob, np.ndarray)
        self.assertEqual(blob.shape, (1, 3, 256, 256))
        self.assertEqual(blob.dtype, np.float32)
        self.assertTrue(np.isfinite(blob).all())

    def test_preprocess_returns_meta_when_requested(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        blob, meta = preprocess_bgr_frame(image, img_size=256, return_meta=True)

        self.assertEqual(blob.shape, (1, 3, 256, 256))
        self.assertEqual(meta["orig_w"], 200.0)
        self.assertEqual(meta["orig_h"], 100.0)
        self.assertAlmostEqual(meta["scale"], 1.28)
        self.assertEqual(meta["pad_x"], 0.0)
        self.assertEqual(meta["pad_y"], 64.0)

    def test_preprocess_uses_imagenet_normalization(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        blob = preprocess_bgr_frame(image, img_size=32)

        expected = (0.0 - IMAGENET_MEAN) / IMAGENET_STD
        np.testing.assert_allclose(blob[0, :, 0, 0], expected.astype(np.float32), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
