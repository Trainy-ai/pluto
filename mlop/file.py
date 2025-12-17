import hashlib
import logging
import mimetypes
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import soundfile as sf
from PIL import Image as PILImage

from .util import get_class

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = 'File'

VALID_CHAR = re.compile(r'^[a-zA-Z0-9_\-.]+$')
INVALID_CHAR = re.compile(r'[^a-zA-Z0-9_\-.]')


class File:
    tag = tag

    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._tmp: Optional[str] = None
        self._path: Optional[str] = os.path.abspath(path)
        self._id: str = self._hash()

        if not name:
            name = self._id
        elif not VALID_CHAR.match(name):
            e = ValueError(
                'invalid file name: '
                f'{name}; file name may only contain alphanumeric characters, '
                'dashes, underscores, and periods; proceeding with sanitized name'
            )
            logger.debug(f'{self.tag}: %s', e)
            name = INVALID_CHAR.sub('-', name)
        self._name = name
        self._ext = os.path.splitext(path)[-1]
        self._type = self._mimetype()
        self._stat = os.stat(path)
        self._url: Optional[str] = None

    def _mimetype(self) -> str:
        if self._path is None:
            return 'application/octet-stream'
        return mimetypes.guess_type(self._path)[0] or 'application/octet-stream'

    def _hash(self) -> str:  # do not truncate
        if self._path is None:
            raise ValueError('File path is not set')
        with open(self._path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _mkcopy(self, dir: str) -> None:
        if not hasattr(self, '_tmp'):
            self._tmp = f'{dir}/files/{self._name}-{self._id}{self._ext}'
            if self._path is None:
                raise ValueError('File path is not set')
            shutil.copyfile(self._path, self._tmp)
            if hasattr(self, '_image'):
                os.remove(self._path)
            self._path = os.path.abspath(self._tmp)


class Artifact(File):
    tag = 'Artifact'

    def __init__(
        self,
        data: Optional[str] = None,
        caption: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._name = caption + f'.{uuid.uuid4()}' if caption else f'{uuid.uuid4()}'
        self._id = f'{uuid.uuid4()}{uuid.uuid4()}'.replace('-', '')

        self._metadata: Dict[str, Any] = metadata or {}
        self._path: Optional[str] = None
        if isinstance(data, str) and os.path.exists(data):
            self._path = os.path.abspath(data)

    # TODO: remove legacy compat
    def add_file(self, path: str, name: Optional[str] = None) -> None:
        self._name = name + f'.{uuid.uuid4()}' if name else f'{uuid.uuid4()}'
        self._path = os.path.abspath(path)

    def load(self, dir: Optional[str] = None) -> None:
        if not self._path:
            logger.critical(f'{self.tag}: failed to load artifact')
        else:
            if self._path is None:
                raise ValueError('Artifact has no path to load')
            super().__init__(path=self._path, name=self._name)


class Text(File):
    tag = 'Text'

    def __init__(self, data: Union[str, Any], caption: Optional[str] = None) -> None:
        self._name = caption + f'.{uuid.uuid4()}' if caption else f'{uuid.uuid4()}'
        self._id = f'{uuid.uuid4()}{uuid.uuid4()}'.replace('-', '')
        self._ext = '.txt'
        self._path: Optional[str] = None

        if isinstance(data, str):
            if os.path.exists(data):
                self._path = os.path.abspath(data)
            else:
                self._text = data
        else:
            self._text = str(data)

    def load(self, dir: Optional[str] = None) -> None:
        if not self._path:
            if dir:
                self._tmp = f'{dir}/files/{self._name}-{self._id}{self._ext}'
                with open(self._tmp, 'w') as f:
                    f.write(self._text)
                self._path = os.path.abspath(self._tmp)

        if self._path is None:
            raise ValueError('Text has no path to load')
        super().__init__(path=self._path, name=self._name)


class Image(File):
    tag = 'Image'

    def __init__(
        self,
        data: Union[str, 'PILImage.Image', np.ndarray],
        caption: Optional[str] = None,
    ) -> None:
        self._name = caption + f'.{uuid.uuid4()}' if caption else f'{uuid.uuid4()}'
        self._id = f'{uuid.uuid4()}{uuid.uuid4()}'.replace('-', '')
        self._ext = '.png'
        self._image: Any = None
        self._path: Optional[str] = None
        self._matplotlib = False

        if isinstance(data, str):
            logger.debug(f'{self.tag}: used file')
            self._image = 'file'  # self._image = PILImage.open(data)
            self._path = os.path.abspath(data)
        else:
            class_name = get_class(data)
            if class_name.startswith('PIL.Image.Image'):
                logger.debug(f'{self.tag}: used PILImage')
                self._image = data
            elif class_name.startswith('matplotlib.'):
                logger.debug(f'{self.tag}: attempted conversion from matplotlib')
                self._matplotlib = True
                self._image = data
            elif class_name.startswith('torch.') and (
                'Tensor' in class_name or 'Variable' in class_name
            ):
                logger.debug(f'{self.tag}: attempted conversion from torch')
                self._image = make_compat_image_torch(data)
            else:
                logger.debug(f'{self.tag}: attempted conversion from array')
                self._image = make_compat_image_numpy(data)

    def load(self, dir: Optional[str] = None) -> None:
        if not self._path:
            if dir:
                self._tmp = f'{dir}/files/{self._name}-{self._id}{self._ext}'
                if getattr(self, '_matplotlib', False):
                    make_compat_image_matplotlib(self._tmp, self._image)
                else:
                    self._image.save(self._tmp, format=self._ext[1:])
                self._path = os.path.abspath(self._tmp)

        if self._path is None:
            raise ValueError('Image has no path to load')
        super().__init__(path=self._path, name=self._name)
        if not self._type.startswith('image/'):
            logger.error(
                f'{self.tag}: proceeding with potentially incompatible mime type: '
                f'{self._type}'
            )


class Audio(File):
    tag = 'Audio'

    def __init__(
        self,
        data: Union[str, np.ndarray],
        rate: Optional[int] = 48000,
        caption: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # TODO: remove legacy compat
        rate = kwargs.get('sample_rate', rate)

        self._name = caption + f'.{uuid.uuid4()}' if caption else f'{uuid.uuid4()}'
        self._id = f'{uuid.uuid4()}{uuid.uuid4()}'.replace('-', '')
        self._ext = '.wav'
        self._audio: Union[str, np.ndarray]
        self._path: Optional[str] = None
        self._rate: int = int(rate) if rate is not None else 48000

        if isinstance(data, str):
            logger.debug(f'{self.tag}: used file')
            self._audio = 'file'
            self._path = os.path.abspath(data)
        else:
            if isinstance(data, np.ndarray):
                logger.debug(f'{self.tag}: used numpy array')
                self._audio = data
            else:
                logger.critical(f'{self.tag}: unsupported data type: %s', type(data))

    def load(self, dir: Optional[str] = None) -> None:
        if not self._path:
            if dir:
                self._tmp = f'{dir}/files/{self._name}-{self._id}{self._ext}'
                sf.write(file=self._tmp, data=self._audio, samplerate=self._rate)
                self._path = os.path.abspath(self._tmp)

        if self._path is None:
            raise ValueError('Audio has no path to load')
        super().__init__(path=self._path, name=self._name)


class Video(File):
    tag = 'Video'

    def __init__(
        self,
        data: Union[str, np.ndarray],
        rate: Optional[int] = 30,
        caption: Optional[str] = None,
        format: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        rate = kwargs.get('fps', rate)

        self._name = caption + f'.{uuid.uuid4()}' if caption else f'{uuid.uuid4()}'
        self._id = f'{uuid.uuid4()}{uuid.uuid4()}'.replace('-', '')
        self._ext = f'.{format}' if format in ['mp4', 'webm', 'ogg', 'gif'] else '.mp4'
        self._path: Optional[str] = None
        self._video: Any = None
        self._data: Optional[np.ndarray] = None
        self._rate: int = int(rate) if rate is not None else 30

        if isinstance(data, str):
            logger.debug(f'{self.tag}: used file')
            self._video = 'file'
            self._path = os.path.abspath(data)
        else:
            if hasattr(data, 'numpy') or isinstance(data, np.ndarray):
                if hasattr(data, 'numpy'):
                    logger.debug(f'{self.tag}: used tensor')
                    self._data = data.numpy()
                else:
                    logger.debug(f'{self.tag}: used numpy array')
                    self._data = data
                if self._data is not None:
                    self._video = make_compat_video_moviepy(self._data, self._rate)
            else:
                logger.critical(f'{self.tag}: unsupported data type: %s', type(data))

    def load(self, dir: Optional[str] = None) -> None:
        if not self._path:
            if dir:
                self._tmp = f'{dir}/files/{self._name}-{self._id}{self._ext}'
                try:
                    make_compat_video_imageio(
                        self._tmp, self._video, self._rate
                    )  # self._video.write_videofile(self._tmp)
                except TypeError as e:
                    Path(self._tmp).touch()
                    logger.critical('%s: failed to write video: %s', self.tag, e)
                self._path = os.path.abspath(self._tmp)

        if self._path is None:
            raise ValueError('Video has no path to load')
        super().__init__(path=self._path, name=self._name)


def make_compat_video_imageio(f: str, clip: Any, rate: int) -> None:
    import imageio

    writer = imageio.save(f, fps=rate)
    for i in clip.iter_frames(fps=rate):
        writer.append_data(i)
    writer.close()


def make_compat_video_moviepy(v: Any, rate: int) -> Any:
    from moviepy.editor import ImageSequenceClip

    t = make_compat_video_numpy(v)
    # _, h, w, c = t.shape
    clip = ImageSequenceClip(list(t), fps=rate)
    return clip


def make_compat_video_numpy(v: Any) -> np.ndarray:
    import numpy as np

    if v.ndim < 4:
        raise ValueError(
            f'{tag}: video data must have at least 4 dimensions: '
            'time, channel, height, width'
        )
    elif v.ndim == 4:
        v = v.reshape(1, *v.shape)
    b, t, c, h, w = v.shape

    if v.dtype != np.uint8:
        logger.debug(f'{tag}: converting video data to uint8')
        v = v.astype(np.uint8)

    rows = 2 ** ((b.bit_length() - 1) // 2)
    cols = v.shape[0] // rows

    v = v.reshape(rows, cols, t, c, h, w)
    v = np.transpose(v, axes=(2, 0, 4, 1, 5, 3))
    v = v.reshape(t, rows * h, cols * w, c)
    return v


def make_compat_image_matplotlib(f: str, val: Any) -> None:
    # from matplotlib.spines import Spine # only required for is_frame_like workaround
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    if val == plt:
        val = val.gcf()
    elif not isinstance(val, Figure):
        if hasattr(val, 'figure'):
            val = val.figure
            if not isinstance(val, Figure):
                e = ValueError(
                    'Invalid matplotlib object; must be a matplotlib.pyplot '
                    'or matplotlib.figure.Figure object'
                )
                logger.critical(f'{tag}: Image conversion failed: %s', e)
                raise e

    val.savefig(f, format='png')


def make_compat_image_torch(val: Any) -> PILImage.Image:
    from torchvision.utils import make_grid

    if hasattr(val, 'requires_grad') and val.requires_grad:
        val = val.detach()
    if hasattr(val, 'dtype') and str(val.dtype).startswith('torch.uint8'):
        val = val.to(float)
    data = make_grid(val, normalize=True)
    image = PILImage.fromarray(
        data.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    )
    return image


def make_compat_image_numpy(val: Any) -> PILImage.Image:
    import numpy as np

    if hasattr(val, 'numpy'):
        val = val.numpy()
    if val.ndim > 2:
        val = val.squeeze()

    if np.min(val) < 0:
        val = (val - np.min(val)) / (np.max(val) - np.min(val))
    if np.max(val) <= 1:
        val = (val * 255).astype(np.int32)
    val = val.clip(0, 255).astype(np.uint8)

    image = PILImage.fromarray(val, mode='RGBA' if val.shape[-1] == 4 else 'RGB')
    return image
