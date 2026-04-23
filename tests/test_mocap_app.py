import importlib.util
from pathlib import Path
import sys
import types
import pytest

try:
    import flask  # noqa: F401
    FLASK_AVAILABLE = True
except ModuleNotFoundError:
    FLASK_AVAILABLE = False


def load_module():
    class FakeArray(list):
        def __sub__(self, other):
            return FakeArray([a - b for a, b in zip(self, other)])

    class FakeLinalg:
        @staticmethod
        def norm(vec):
            return sum(v * v for v in vec) ** 0.5

    fake_np = types.SimpleNamespace(
        array=lambda values, dtype=None: FakeArray(values),
        dot=lambda a, b: sum(x * y for x, y in zip(a, b)),
        linalg=FakeLinalg(),
        degrees=lambda rad: rad * (180.0 / 3.141592653589793),
        arccos=lambda val: __import__("math").acos(val),
        clip=lambda val, low, high: max(low, min(high, val)),
        gradient=lambda arr, *_args, **_kwargs: arr,
        float64=float,
    )

    class FakePose:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def process(self, _image):
            return types.SimpleNamespace(pose_landmarks=None)

    fake_cv2 = types.SimpleNamespace(
        VideoCapture=lambda *_args, **_kwargs: types.SimpleNamespace(isOpened=lambda: False, release=lambda: None),
        cvtColor=lambda image, _code: image,
        COLOR_BGR2RGB=1,
        COLOR_RGB2BGR=2,
        imencode=lambda *_args, **_kwargs: (True, types.SimpleNamespace(tobytes=lambda: b"")),
    )
    fake_mp_pose = types.SimpleNamespace(
        Pose=lambda **_kwargs: FakePose(),
        PoseLandmark=types.SimpleNamespace(
            LEFT_SHOULDER=types.SimpleNamespace(value=0),
            LEFT_ELBOW=types.SimpleNamespace(value=1),
            LEFT_WRIST=types.SimpleNamespace(value=2),
            LEFT_HIP=types.SimpleNamespace(value=3),
            LEFT_KNEE=types.SimpleNamespace(value=4),
            LEFT_ANKLE=types.SimpleNamespace(value=5),
        ),
        POSE_CONNECTIONS=[],
    )
    fake_mp = types.SimpleNamespace(
        solutions=types.SimpleNamespace(
            drawing_utils=types.SimpleNamespace(draw_landmarks=lambda *_args, **_kwargs: None),
            pose=fake_mp_pose,
        )
    )
    fake_plt = types.SimpleNamespace(
        subplots=lambda *_args, **_kwargs: (
            object(),
            [
                types.SimpleNamespace(
                    plot=lambda *_a, **_k: None,
                    set_title=lambda *_a, **_k: None,
                    set_xlabel=lambda *_a, **_k: None,
                    set_ylabel=lambda *_a, **_k: None,
                    legend=lambda *_a, **_k: None,
                    grid=lambda *_a, **_k: None,
                ),
                types.SimpleNamespace(
                    plot=lambda *_a, **_k: None,
                    set_title=lambda *_a, **_k: None,
                    set_xlabel=lambda *_a, **_k: None,
                    set_ylabel=lambda *_a, **_k: None,
                    legend=lambda *_a, **_k: None,
                    grid=lambda *_a, **_k: None,
                ),
            ],
        ),
        tight_layout=lambda: None,
        savefig=lambda *_args, **_kwargs: None,
        close=lambda *_args, **_kwargs: None,
    )
    fake_matplotlib = types.SimpleNamespace(pyplot=fake_plt)

    sys.modules.setdefault("numpy", fake_np)
    sys.modules.setdefault("cv2", fake_cv2)
    sys.modules.setdefault("mediapipe", fake_mp)
    sys.modules.setdefault("matplotlib", fake_matplotlib)
    sys.modules.setdefault("matplotlib.pyplot", fake_plt)

    module_path = Path(__file__).resolve().parents[1] / "Media Pipe Pose Tutorial.py"
    spec = importlib.util.spec_from_file_location("mocap_app", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_calculate_angle_right_angle():
    if not FLASK_AVAILABLE:
        pytest.skip("flask is not installed in this environment")
    module = load_module()
    angle = module.calculate_angle([0, 1], [0, 0], [1, 0])
    assert abs(angle - 90.0) < 1e-6


def test_generate_velocity_graph_requires_data():
    if not FLASK_AVAILABLE:
        pytest.skip("flask is not installed in this environment")
    module = load_module()
    client = module.app.test_client()

    with module.data_lock:
        module.data_log.clear()

    response = client.post("/generate_velocity_graph")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_reset_session_clears_data():
    if not FLASK_AVAILABLE:
        pytest.skip("flask is not installed in this environment")
    module = load_module()
    client = module.app.test_client()

    with module.data_lock:
        module.data_log.extend([[1.0, 10.0, 20.0, 30.0, 40.0], [2.0, 11.0, 21.0, 31.0, 41.0]])

    response = client.post("/reset_session")
    assert response.status_code == 200

    with module.data_lock:
        assert module.data_log == []
