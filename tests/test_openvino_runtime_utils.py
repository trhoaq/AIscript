import unittest

from core.openvino_runtime_utils import get_openvino_input_name, resolve_square_input_size


class _DummyPort:
    def __init__(self, any_name=None, names=None, raise_any=False, shape=None):
        self._any_name = any_name
        self._names = names or []
        self._raise_any = raise_any
        self.shape = shape or [1, 3, 320, 320]

    def get_any_name(self):
        if self._raise_any:
            raise RuntimeError("no any name")
        return self._any_name

    def get_names(self):
        return self._names


class _DummyModel:
    def __init__(self, port, use_inputs_attr=False, raise_input=False):
        self._port = port
        self._raise_input = raise_input
        self.inputs = [port] if use_inputs_attr else []

    def input(self, _idx):
        if self._raise_input:
            raise RuntimeError("input() unavailable")
        return self._port


class OpenVinoRuntimeUtilsTests(unittest.TestCase):
    def test_get_openvino_input_name_prefers_any_name(self):
        model = _DummyModel(_DummyPort(any_name="images_tensor", names=["fallback"]))
        self.assertEqual(get_openvino_input_name(model), "images_tensor")

    def test_get_openvino_input_name_falls_back_to_names(self):
        model = _DummyModel(_DummyPort(any_name=None, names=["input_0"], raise_any=True))
        self.assertEqual(get_openvino_input_name(model), "input_0")

    def test_get_openvino_input_name_falls_back_to_default(self):
        model = _DummyModel(_DummyPort(any_name=None, names=[], raise_any=True))
        self.assertEqual(get_openvino_input_name(model), "images")

    def test_get_openvino_input_name_uses_inputs_attr_when_input_method_fails(self):
        port = _DummyPort(any_name="from_inputs")
        model = _DummyModel(port, use_inputs_attr=True, raise_input=True)
        self.assertEqual(get_openvino_input_name(model), "from_inputs")

    def test_resolve_square_input_size(self):
        model = _DummyModel(_DummyPort(shape=[1, 3, 256, 256]))
        self.assertEqual(resolve_square_input_size(model, fallback_size=320), 256)

    def test_resolve_square_input_size_fallbacks_on_non_square(self):
        model = _DummyModel(_DummyPort(shape=[1, 3, 256, 192]))
        self.assertEqual(resolve_square_input_size(model, fallback_size=320), 320)

    def test_resolve_square_input_size_fallbacks_on_error(self):
        model = _DummyModel(_DummyPort(), raise_input=True, use_inputs_attr=True)
        self.assertEqual(resolve_square_input_size(model, fallback_size=320), 320)


if __name__ == "__main__":
    unittest.main()
