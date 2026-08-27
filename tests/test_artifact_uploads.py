"""Tests for artifact and file uploads to production.

These tests verify file types not covered by test_basic.py can be properly
uploaded to the production server without HTTP errors.

Covered in test_basic.py (not duplicated here):
- Image: file path, PIL, numpy, matplotlib, torch
- Video: file path, numpy
- Audio: downloaded from URL
"""

import hashlib
import importlib.util
import json
import os

import numpy as np
import pytest
from PIL import Image as PILImage

import pluto
from tests.utils import (
    assert_sync_files_completed,
    capture_sync_db_path,
    download_file_with_poll,
    get_task_name,
)

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    HAS_TORCH = False

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:  # pragma: no cover - optional dependency
    sf = None
    HAS_SOUNDFILE = False

HAS_IMAGEIO = importlib.util.find_spec('imageio') is not None
HAS_MOVIEPY = importlib.util.find_spec('moviepy') is not None
HAS_VIDEO_DEPS = HAS_IMAGEIO and HAS_MOVIEPY

TESTING_PROJECT_NAME = 'testing-ci'


@pytest.fixture
def pluto_run():
    """Initializes and finishes a pluto run for a test.

    After finish(), asserts every file the test logged actually uploaded
    (sync DB rows all COMPLETED) — otherwise these tests are green even
    when payloads are silently lost.
    """
    run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config={})
    yield run
    db_path = capture_sync_db_path(run)
    run.finish()
    assert_sync_files_completed(db_path)


class TestArtifactUploads:
    """Test suite for Artifact uploads (not covered in test_basic.py)."""

    def test_artifact_upload_from_file_path(self, tmp_path, pluto_run):
        """Test uploading an Artifact from a file path."""
        artifact_path = tmp_path / 'test_artifact.json'
        artifact_path.write_text(json.dumps({'test': 'data', 'value': 123}))

        artifact = pluto.Artifact(str(artifact_path), caption='json-artifact')
        pluto_run.log({'artifact/json': artifact})

    def test_artifact_upload_binary_file(self, tmp_path, pluto_run):
        """Test uploading a binary artifact."""
        binary_path = tmp_path / 'test_binary.bin'
        binary_path.write_bytes(os.urandom(1024))  # 1KB random data

        artifact = pluto.Artifact(str(binary_path), caption='binary-artifact')
        pluto_run.log({'artifact/binary': artifact})

    def test_artifact_with_metadata(self, tmp_path, pluto_run):
        """Test uploading an artifact with metadata."""
        artifact_path = tmp_path / 'model_weights.pt'
        artifact_path.write_bytes(os.urandom(2048))

        artifact = pluto.Artifact(
            str(artifact_path),
            caption='model-weights',
            metadata={'version': '1.0', 'accuracy': 0.95, 'epochs': 100},
        )
        pluto_run.log({'artifact/model': artifact})


class TestFileUploads:
    """Test suite for generic File uploads (not covered in test_basic.py)."""

    def test_file_upload_text_file(self, tmp_path, pluto_run):
        """Test uploading a generic text file using File class."""
        file_path = tmp_path / 'readme.txt'
        file_path.write_text('This is a test readme file.\nWith multiple lines.')

        file_obj = pluto.File(str(file_path), name='readme')
        pluto_run.log({'file/readme': file_obj})

    def test_file_upload_csv(self, tmp_path, pluto_run):
        """Test uploading a CSV file."""
        csv_path = tmp_path / 'data.csv'
        csv_path.write_text('col1,col2,col3\n1,2,3\n4,5,6\n7,8,9')

        file_obj = pluto.File(str(csv_path), name='data-csv')
        pluto_run.log({'file/csv': file_obj})

    def test_file_upload_large_file(self, tmp_path, pluto_run):
        """Test uploading a larger file (100KB)."""
        large_file_path = tmp_path / 'large_file.bin'
        large_file_path.write_bytes(os.urandom(100 * 1024))  # 100KB

        file_obj = pluto.File(str(large_file_path), name='large-file')
        pluto_run.log({'file/large': file_obj})


class TestTextUploads:
    """Test suite for Text uploads (not covered in test_basic.py)."""

    def test_text_upload_from_string(self, pluto_run):
        """Test uploading Text from a string."""
        text_content = 'This is a test log message.\nWith multiple lines.'
        text = pluto.Text(text_content, caption='log-message')
        pluto_run.log({'text/log': text})

    def test_text_upload_from_file_path(self, tmp_path, pluto_run):
        """Test uploading Text from a file path."""
        text_path = tmp_path / 'log.txt'
        text_path.write_text('Log entry 1\nLog entry 2\nLog entry 3')

        text = pluto.Text(str(text_path), caption='log-file')
        pluto_run.log({'text/file': text})

    def test_text_upload_long_content(self, pluto_run):
        """Test uploading a longer text content."""
        long_text = '\n'.join([f'Line {i}: ' + 'x' * 100 for i in range(100)])
        text = pluto.Text(long_text, caption='long-text')
        pluto_run.log({'text/long': text})

    def test_text_upload_unicode(self, pluto_run):
        """Test uploading text with unicode characters."""
        unicode_text = 'Hello World! 中文 日本語 한국어 😀🚀'
        text = pluto.Text(unicode_text, caption='unicode-text')
        pluto_run.log({'text/unicode': text})


class TestImageUploadsExtended:
    """Extended Image upload tests (beyond test_basic.py coverage)."""

    def test_image_upload_rgba(self, pluto_run):
        """Test uploading an RGBA image."""
        rgba_img = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)

        image = pluto.Image(rgba_img, caption='rgba-image')
        pluto_run.log({'image/rgba': image})

    def test_image_upload_multiple(self, pluto_run):
        """Test uploading multiple images in separate log calls."""
        images = [
            pluto.Image(
                PILImage.new('RGB', (32, 32), color=(i * 80, i * 80, 255 - i * 80)),
                caption=f'multi-image-{i}',
            )
            for i in range(3)
        ]

        for i, img in enumerate(images):
            pluto_run.log({f'image/multi/{i}': img})


class TestVideoUploadsExtended:
    """Extended Video upload tests (beyond test_basic.py coverage)."""

    @pytest.mark.skipif(
        not HAS_VIDEO_DEPS or not HAS_TORCH, reason='video or torch deps not installed'
    )
    def test_video_upload_from_torch_tensor(self, pluto_run):
        """Test uploading a Video from a PyTorch tensor."""
        video_tensor = torch.randint(0, 255, (4, 3, 32, 32), dtype=torch.uint8)

        video = pluto.Video(video_tensor, rate=10, caption='torch-video')
        pluto_run.log({'video/torch': video})

    def test_video_upload_gif_format(self, tmp_path, pluto_run):
        """Test uploading a video with GIF format."""
        gif_path = tmp_path / 'test.gif'
        gif_path.write_bytes(b'GIF89a\x01\x00\x01\x00\x00\x00\x00;\x00')

        video = pluto.Video(str(gif_path), format='gif', caption='gif-video')
        pluto_run.log({'video/gif': video})


class TestAudioUploadsExtended:
    """Extended Audio upload tests (beyond test_basic.py coverage)."""

    def test_audio_upload_from_file_path(self, tmp_path, pluto_run):
        """Test uploading Audio from a generated file path."""
        audio_path = tmp_path / 'test_audio.wav'
        if HAS_SOUNDFILE:
            sample_rate = 22050
            duration = 0.5
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            sf.write(str(audio_path), audio_data, sample_rate)
        else:
            audio_path.write_bytes(
                b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00'
                b'\x01\x00\x01\x00"V\x00\x00"V\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00'
            )

        audio = pluto.Audio(str(audio_path), caption='file-audio')
        pluto_run.log({'audio/file': audio})

    @pytest.mark.skipif(not HAS_SOUNDFILE, reason='soundfile not installed')
    def test_audio_upload_from_numpy(self, pluto_run):
        """Test uploading Audio from a numpy array."""
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        audio = pluto.Audio(audio_data, rate=sample_rate, caption='numpy-audio')
        pluto_run.log({'audio/numpy': audio})

    @pytest.mark.skipif(not HAS_SOUNDFILE, reason='soundfile not installed')
    def test_audio_upload_stereo(self, pluto_run):
        """Test uploading stereo audio."""
        sample_rate = 22050
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
        stereo_data = np.column_stack([left, right])

        audio = pluto.Audio(stereo_data, rate=sample_rate, caption='stereo-audio')
        pluto_run.log({'audio/stereo': audio})


class TestMultipleFileTypesUpload:
    """Test uploading multiple file types in a single run."""

    def test_mixed_file_types_upload(self, tmp_path, pluto_run):
        """Test uploading multiple different file types in a single run."""
        # Text
        text = pluto.Text('Test log content', caption='mixed-text')
        pluto_run.log({'mixed/text': text})

        # Image
        img = PILImage.new('RGB', (32, 32), color='blue')
        image = pluto.Image(img, caption='mixed-image')
        pluto_run.log({'mixed/image': image})

        # Artifact
        artifact_path = tmp_path / 'config.json'
        artifact_path.write_text('{"setting": "value"}')
        artifact = pluto.Artifact(str(artifact_path), caption='mixed-artifact')
        pluto_run.log({'mixed/artifact': artifact})

        # File
        file_path = tmp_path / 'data.txt'
        file_path.write_text('Some data content')
        file_obj = pluto.File(str(file_path), name='mixed-file')
        pluto_run.log({'mixed/file': file_obj})

    def test_sequential_file_uploads(self, pluto_run):
        """Test uploading files sequentially across multiple steps."""
        for step in range(5):
            pluto_run.log({'train/loss': 1.0 / (step + 1)}, step=step)

            img = PILImage.new('RGB', (32, 32), color=(step * 50, 100, 255 - step * 50))
            image = pluto.Image(img, caption=f'step-{step}-image')
            pluto_run.log({'train/image': image}, step=step)

            text = pluto.Text(f'Step {step} completed', caption=f'step-{step}-log')
            pluto_run.log({'train/log': text}, step=step)

    def test_batch_image_upload(self, pluto_run):
        """Test uploading a batch of images."""
        images = [
            pluto.Image(
                PILImage.new('RGB', (16, 16), color=(i * 25, i * 25, i * 25)),
                caption=f'batch-{i}',
            )
            for i in range(10)
        ]

        pluto_run.log({'batch/images': images})


class TestEdgeCases:
    """Test edge cases for file uploads."""

    def test_empty_text_upload(self, pluto_run):
        """Test uploading empty text."""
        text = pluto.Text('', caption='empty-text')
        pluto_run.log({'edge/empty-text': text})

    def test_small_image_upload(self, pluto_run):
        """Test uploading a very small image (1x1)."""
        img = PILImage.new('RGB', (1, 1), color='white')
        image = pluto.Image(img, caption='tiny-image')
        pluto_run.log({'edge/tiny-image': image})

    def test_large_image_upload(self, pluto_run):
        """Test uploading a larger image (512x512)."""
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = pluto.Image(img, caption='large-image')
        pluto_run.log({'edge/large-image': image})

    def test_special_characters_in_caption(self, tmp_path, pluto_run):
        """Test uploading files with special characters in captions."""
        file_path = tmp_path / 'test.txt'
        file_path.write_text('test content')

        artifact = pluto.Artifact(str(file_path), caption='test_artifact-v1.0')
        pluto_run.log({'edge/special-caption': artifact})

    def test_multiple_files_same_key(self, pluto_run):
        """Test uploading multiple files with the same log key across steps."""
        for step in range(3):
            img = PILImage.new('RGB', (16, 16), color=(step * 80, 0, 0))
            image = pluto.Image(img, caption=f'overwrite-{step}')
            pluto_run.log({'same-key/image': image}, step=step)


class TestRoundTripVerification:
    """Upload → finish → query back → verify content actually matches.

    The smoke tests above prove log() doesn't raise; the fixture's sync-DB
    check proves the payload left the client. These tests close the loop
    end-to-end: the bytes the server hands back are the bytes we logged.
    One representative test per file type (images additionally covered by
    test_e2e.py's pixel checks).
    """

    def _finished_run(self):
        run = pluto.init(project=TESTING_PROJECT_NAME, name=get_task_name(), config={})
        return run, run.settings._op_id

    def test_artifact_bytes_round_trip(self, tmp_path):
        """Binary artifact downloads back byte-identical."""
        payload = os.urandom(4096)
        source = tmp_path / 'weights.bin'
        source.write_bytes(payload)

        run, run_id = self._finished_run()
        run.log({'roundtrip/artifact': pluto.Artifact(str(source), caption='rt-bin')})
        run.finish()

        path = download_file_with_poll(
            TESTING_PROJECT_NAME, run_id, 'roundtrip/artifact', tmp_path / 'dl'
        )
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == hashlib.sha256(payload).hexdigest()
        ), 'downloaded artifact bytes differ from uploaded bytes'

    def test_file_bytes_round_trip(self, tmp_path):
        """Generic File downloads back byte-identical."""
        content = 'col1,col2\n1,2\n3,4\n'
        source = tmp_path / 'data.csv'
        source.write_text(content)

        run, run_id = self._finished_run()
        run.log({'roundtrip/file': pluto.File(str(source), name='rt-csv')})
        run.finish()

        path = download_file_with_poll(
            TESTING_PROJECT_NAME, run_id, 'roundtrip/file', tmp_path / 'dl'
        )
        assert (
            path.read_bytes() == content.encode()
        ), 'downloaded file content differs from uploaded content'

    def test_text_round_trip(self, tmp_path):
        """Text logged from a string downloads back with identical content."""
        content = f'round-trip sentinel {get_task_name()}\nline two\n'

        run, run_id = self._finished_run()
        run.log({'roundtrip/text': pluto.Text(content, caption='rt-text')})
        run.finish()

        path = download_file_with_poll(
            TESTING_PROJECT_NAME, run_id, 'roundtrip/text', tmp_path / 'dl'
        )
        assert (
            path.read_text() == content
        ), f'downloaded text differs: {path.read_text()!r} != {content!r}'

    def test_image_pixels_round_trip(self, tmp_path):
        """Image downloads back pixel-identical (PNG is lossless)."""
        source_img = PILImage.new('RGB', (2, 2))
        pixels = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        source_img.putdata(pixels)

        run, run_id = self._finished_run()
        run.log({'roundtrip/image': pluto.Image(source_img, caption='rt-img')})
        run.finish()

        path = download_file_with_poll(
            TESTING_PROJECT_NAME, run_id, 'roundtrip/image', tmp_path / 'dl'
        )
        downloaded = PILImage.open(path).convert('RGB')
        assert downloaded.size == (2, 2)
        assert (
            list(downloaded.getdata()) == pixels
        ), 'downloaded image pixels differ from uploaded image'
