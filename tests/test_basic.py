import importlib.util

import pytest
from PIL import Image as PILImage

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover - optional dependency
    matplotlib = None
    plt = None
    HAS_MATPLOTLIB = False

import numpy as np

import mlop
from tests.utils import get_task_name

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    HAS_TORCH = False

HAS_IMAGEIO = importlib.util.find_spec('imageio') is not None
HAS_MOVIEPY = importlib.util.find_spec('moviepy') is not None
HAS_VIDEO_DEPS = HAS_IMAGEIO and HAS_MOVIEPY

TESTING_PROJECT_NAME = 'testing-ci'


def test_quickstart():
    task_name = get_task_name()
    config = {'lr': 0.001, 'epochs': 100}
    run = mlop.init(project=TESTING_PROJECT_NAME, name=task_name, config=config)
    for i in range(config['epochs']):
        run.log({'val/loss': 0})
        run.log({'val/x': i})
        print(i)
    run.finish()


def test_init_with_hyperparameters():
    task_name = get_task_name()
    config = {'lr': 0.01, 'batch_size': 32, 'epochs': 50}
    run = mlop.init(project=TESTING_PROJECT_NAME, name=task_name, config=config)
    assert run.config['lr'] == 0.01
    assert run.config['batch_size'] == 32
    assert run.config['epochs'] == 50
    run.finish()


def _log_image(image, key, task_name):
    run = mlop.init(project=TESTING_PROJECT_NAME, name=task_name, config={})
    run.log({key: image})
    run.finish()


def _log_video(video, key, task_name):
    run = mlop.init(project=TESTING_PROJECT_NAME, name=task_name, config={})
    run.log({key: video})
    run.finish()


def test_image_logging_from_file_path(tmp_path):
    img_path = tmp_path / 'example.png'
    PILImage.new('RGB', (4, 4), color='white').save(img_path)
    image = mlop.Image(str(img_path), caption='file-path')
    _log_image(image, 'image/file/path', get_task_name())


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib not installed')
def test_image_logging_from_pil_image():
    pil_img = PILImage.new('RGB', (4, 4), color='blue')
    image = mlop.Image(pil_img, caption='pil-image')
    _log_image(image, 'image/pil', get_task_name())


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib not installed')
def test_image_logging_from_numpy_array():
    np_img = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
    image = mlop.Image(np_img, caption='numpy-array')
    _log_image(image, 'image/numpy', get_task_name())


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib not installed')
def test_image_logging_from_matplotlib_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.axis('off')
    image = mlop.Image(fig, caption='matplotlib-fig')
    _log_image(image, 'image/matplotlib', get_task_name())
    plt.close(fig)


@pytest.mark.skipif(not HAS_TORCH, reason='torch not installed')
def test_image_logging_from_torch_tensor():
    pytest.importorskip('torchvision.utils')
    tensor = torch.rand(3, 4, 4)
    image = mlop.Image(tensor, caption='torch-tensor')
    _log_image(image, 'image/torch', get_task_name())


def test_video_logging_from_file_path(tmp_path):
    video_path = tmp_path / 'sample.mp4'
    video_path.write_bytes(b'\x00')
    video = mlop.Video(str(video_path), caption='video-file')
    _log_video(video, 'video/file/path', get_task_name())


@pytest.mark.skipif(not HAS_VIDEO_DEPS, reason='video dependencies not installed')
def test_video_logging_from_numpy_array():
    video_array = np.random.randint(0, 255, (2, 3, 4, 4), dtype=np.uint8)
    video = mlop.Video(video_array, rate=5, caption='video-numpy')
    _log_video(video, 'video/numpy', get_task_name())

