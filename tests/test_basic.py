import importlib.util

import httpx
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

import pluto
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
    run = pluto.init(project=TESTING_PROJECT_NAME, name=task_name, config=config)
    for i in range(config['epochs']):
        run.log({'val/loss': 0})
        run.log({'val/x': i})
        print(i)
    run.finish()


def test_init_with_hyperparameters():
    task_name = get_task_name()
    config = {'lr': 0.01, 'batch_size': 32, 'epochs': 50}
    run = pluto.init(project=TESTING_PROJECT_NAME, name=task_name, config=config)
    assert run.config['lr'] == 0.01
    assert run.config['batch_size'] == 32
    assert run.config['epochs'] == 50
    run.finish()


def _log_image(image, key, task_name):
    run = pluto.init(project=TESTING_PROJECT_NAME, name=task_name, config={})
    run.log({key: image})
    run.finish()


def _log_video(video, key, task_name):
    run = pluto.init(project=TESTING_PROJECT_NAME, name=task_name, config={})
    run.log({key: video})
    run.finish()


def test_image_logging_from_file_path(tmp_path):
    img_path = tmp_path / 'example.png'
    PILImage.new('RGB', (4, 4), color='white').save(img_path)
    image = pluto.Image(str(img_path), caption='file-path')
    _log_image(image, 'image/file/path', get_task_name())


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib not installed')
def test_image_logging_from_pil_image():
    pil_img = PILImage.new('RGB', (4, 4), color='blue')
    image = pluto.Image(pil_img, caption='pil-image')
    _log_image(image, 'image/pil', get_task_name())


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib not installed')
def test_image_logging_from_numpy_array():
    np_img = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
    image = pluto.Image(np_img, caption='numpy-array')
    _log_image(image, 'image/numpy', get_task_name())


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib not installed')
def test_image_logging_from_matplotlib_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.axis('off')
    image = pluto.Image(fig, caption='matplotlib-fig')
    _log_image(image, 'image/matplotlib', get_task_name())
    plt.close(fig)


@pytest.mark.skipif(not HAS_TORCH, reason='torch not installed')
def test_image_logging_from_torch_tensor():
    pytest.importorskip('torchvision.utils')
    tensor = torch.rand(3, 4, 4)
    image = pluto.Image(tensor, caption='torch-tensor')
    _log_image(image, 'image/torch', get_task_name())


def test_video_logging_from_file_path(tmp_path):
    video_path = tmp_path / 'sample.mp4'
    video_path.write_bytes(b'\x00')
    video = pluto.Video(str(video_path), caption='video-file')
    _log_video(video, 'video/file/path', get_task_name())


@pytest.mark.skipif(not HAS_VIDEO_DEPS, reason='video dependencies not installed')
def test_video_logging_from_numpy_array():
    video_array = np.random.randint(0, 255, (2, 3, 4, 4), dtype=np.uint8)
    video = pluto.Video(video_array, rate=5, caption='video-numpy')
    _log_video(video, 'video/numpy', get_task_name())


def test_audio_logging_from_downloaded_file(tmp_path):
    url = 'https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg'
    audio_path = tmp_path / 'test.ogg'
    try:
        response = httpx.get(url, timeout=10)
        response.raise_for_status()
    except httpx.HTTPError as exc:  # pragma: no cover - network issues
        pytest.skip(f'Audio download failed: {exc}')

    audio_path.write_bytes(response.content)
    audio = pluto.Audio(str(audio_path))
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config={})
    run.log({'audio': audio})
    run.finish()


def test_histogram_logging_from_numpy_array():
    data = np.random.normal(loc=0.0, scale=1.0, size=1000)
    histogram = pluto.Histogram(data=data, bins=32)
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config={})
    run.log({'metrics/histogram': histogram})
    assert histogram.to_dict()['shape'] == 'uniform'
    run.finish()


def test_tags_initialization_with_string():
    """Test initializing a run with a single tag string."""
    run = pluto.init(
        project=TESTING_PROJECT_NAME, name=get_task_name(), tags='experiment'
    )
    assert 'experiment' in run.tags
    assert len(run.tags) == 1
    run.finish()


def test_tags_initialization_with_list():
    """Test initializing a run with multiple tags."""
    run = pluto.init(
        project=TESTING_PROJECT_NAME,
        name=get_task_name(),
        tags=['production', 'v2', 'baseline'],
    )
    assert 'production' in run.tags
    assert 'v2' in run.tags
    assert 'baseline' in run.tags
    assert len(run.tags) == 3
    run.finish()


def test_add_tags_single_string():
    """Test adding a single tag as a string."""
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name())
    assert len(run.tags) == 0

    run.add_tags('experiment')
    assert 'experiment' in run.tags
    assert len(run.tags) == 1

    run.finish()


def test_add_tags_list():
    """Test adding multiple tags as a list."""
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), tags='initial')
    assert len(run.tags) == 1

    run.add_tags(['production', 'v2'])
    assert 'initial' in run.tags
    assert 'production' in run.tags
    assert 'v2' in run.tags
    assert len(run.tags) == 3

    run.finish()


def test_add_tags_duplicate():
    """Test that adding duplicate tags doesn't create duplicates."""
    run = pluto.init(
        project=TESTING_PROJECT_NAME, name=get_task_name(), tags=['experiment']
    )
    assert len(run.tags) == 1

    run.add_tags('experiment')  # Try to add same tag
    assert len(run.tags) == 1  # Should still be 1

    run.add_tags(['experiment', 'new-tag'])
    assert len(run.tags) == 2  # Should add only 'new-tag'
    assert 'experiment' in run.tags
    assert 'new-tag' in run.tags

    run.finish()


def test_remove_tags_single_string():
    """Test removing a single tag."""
    run = pluto.init(
        project=TESTING_PROJECT_NAME, name=get_task_name(), tags=['exp', 'prod', 'v1']
    )
    assert len(run.tags) == 3

    run.remove_tags('exp')
    assert 'exp' not in run.tags
    assert len(run.tags) == 2

    run.finish()


def test_remove_tags_list():
    """Test removing multiple tags."""
    run = pluto.init(
        project=TESTING_PROJECT_NAME,
        name=get_task_name(),
        tags=['exp', 'prod', 'v1', 'baseline'],
    )
    assert len(run.tags) == 4

    run.remove_tags(['exp', 'v1'])
    assert 'exp' not in run.tags
    assert 'v1' not in run.tags
    assert 'prod' in run.tags
    assert 'baseline' in run.tags
    assert len(run.tags) == 2

    run.finish()


def test_remove_tags_nonexistent():
    """Test that removing nonexistent tags doesn't cause errors."""
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), tags=['exp'])
    assert len(run.tags) == 1

    run.remove_tags('nonexistent')  # Should not raise error
    assert len(run.tags) == 1

    run.remove_tags(['exp', 'also-nonexistent'])
    assert len(run.tags) == 0  # Only 'exp' should be removed

    run.finish()


def test_update_config_basic():
    """Test updating config after run initialization."""
    run = pluto.init(
        project=TESTING_PROJECT_NAME, name=get_task_name(), config={'lr': 0.001}
    )
    assert run.config['lr'] == 0.001

    run.update_config({'epochs': 100, 'model': 'resnet50'})
    assert run.config['lr'] == 0.001  # Original preserved
    assert run.config['epochs'] == 100
    assert run.config['model'] == 'resnet50'

    run.finish()


def test_update_config_override():
    """Test that update_config overrides existing keys."""
    run = pluto.init(
        project=TESTING_PROJECT_NAME,
        name=get_task_name(),
        config={'lr': 0.001, 'batch_size': 32},
    )
    assert run.config['lr'] == 0.001

    run.update_config({'lr': 0.01})  # Override lr
    assert run.config['lr'] == 0.01
    assert run.config['batch_size'] == 32  # Unchanged

    run.finish()


def test_update_config_on_empty():
    """Test update_config when initial config is empty."""
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config={})
    assert run.config == {}

    run.update_config({'model': 'gpt-4', 'temperature': 0.7})
    assert run.config['model'] == 'gpt-4'
    assert run.config['temperature'] == 0.7

    run.finish()


def test_update_config_multiple_calls():
    """Test multiple update_config calls accumulate correctly."""
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config={})

    run.update_config({'lr': 0.001})
    run.update_config({'epochs': 100})
    run.update_config({'model': 'resnet50', 'lr': 0.01})  # Override lr

    assert run.config['lr'] == 0.01
    assert run.config['epochs'] == 100
    assert run.config['model'] == 'resnet50'

    run.finish()
