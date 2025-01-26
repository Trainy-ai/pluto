import hashlib
import logging
import mimetypes
import os
import re
import shutil
import tempfile

from PIL import Image as PILImage

from .util import get_class

logger = logging.getLogger(f"{__name__.split('.')[0]}")


class File:
    def __init__(
        self,
        path: str,
        name: str | None = None,
    ) -> None:
        self._path = path
        self._id = self._hash()

        if not name:
            name = self._id
        elif not re.match(r"^[a-zA-Z0-9_\-.]+$", name):
            e = ValueError(
                f"invalid file name: {name}; file name must only contain alphanumeric characters, dashes, underscores, and periods."
            )
            logger.warning("File: %s", e)
            name = re.sub(r"[^a-zA-Z0-9_\-.]", "-", name)  # raise e
        self._name = name
        self._type = self._mimetype()
        self._stat = os.stat(self._path)
        self._size = self._stat.st_size  # os.path.getsize(self._path)
        self._ext = os.path.splitext(self._path)[-1]
        self._url = None

    def _mimetype(self) -> str:
        return mimetypes.guess_type(self._path)[0] or "application/octet-stream"

    def _hash(self) -> str:  # do not truncate
        return hashlib.sha256(self._path.encode()).hexdigest()

    def _mkcopy(self, name, work_dir) -> None:
        # self._name = name
        self._tmp = f"{work_dir}/files/{self._name}-{self._id}{self._ext}"
        os.makedirs(os.path.dirname(self._tmp), exist_ok=True)
        shutil.copyfile(self._path, self._tmp)
        if hasattr(self, "_image"):
            os.remove(self._path)
        self._path = self._tmp


class Image(File):
    def __init__(
        self,
        data: any,  # Union[PILImage.Image, np.ndarray],
        caption: str | None = None,
    ) -> None:
        if isinstance(data, str):
            logger.debug("Image: used PILImage from path")
            self._image = "file"  # self._image = PILImage.open(data)
            path = os.path.abspath(data)
        else:
            if isinstance(data, PILImage.Image):
                logger.debug("Image: used PILImage")
                self._image = data
            else:
                class_name = get_class(data)
                if class_name.startswith("matplotlib."):
                    logger.debug("Image: attempted conversion from matplotlib")
                    self._image = make_compat_matplotlib(data)
                elif class_name.startswith("torch.") and (
                    "Tensor" in class_name or "Variable" in class_name
                ):
                    logger.debug("Image: attempted conversion from torch")
                    self._image = make_compat_torch(data)
                else:
                    logger.debug("Image: attempted conversion from array")
                    self._image = make_compat_numpy(data)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".PNG") as tmp:
                self._image.save(tmp.name, format="PNG")
                path = os.path.abspath(tmp.name)

        super().__init__(path=path, name=caption)
        if not self._type.startswith("image/"):
            logger.error(
                f"Image: proceeding with potentially incompatible mime type: {self._type}"
            )


def make_compat_matplotlib(val: any) -> any:
    # from matplotlib.spines import Spine # only required for is_frame_like workaround
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    if val == plt:
        val = val.gcf()
    elif not isinstance(val, Figure):
        if hasattr(val, "figure"):
            val = val.figure
            if not isinstance(val, Figure):
                e = ValueError(
                    "Invalid matplotlib object; must be a matplotlib.pyplot or matplotlib.figure.Figure object."
                )
                logger.critical("Image failed: %s", e)
                raise e

    from io import BytesIO

    buf = BytesIO()
    val.savefig(buf, format="png")
    image = PILImage.open(buf, formats=["PNG"])
    return image


def make_compat_torch(val: any) -> any:
    from torchvision.utils import make_grid

    if hasattr(val, "requires_grad") and val.requires_grad:
        val = val.detach()
    if hasattr(val, "dtype") and str(val.dtype).startswith("torch.uint8"):
        val = val.to(float)
    data = make_grid(val, normalize=True)
    image = PILImage.fromarray(
        data.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    )
    return image


def make_compat_numpy(val: any) -> any:
    import numpy as np

    if hasattr(val, "numpy"):
        val = val.numpy()
    if val.ndim > 2:
        val = val.squeeze()

    if np.min(val) < 0:
        val = (val - np.min(val)) / (np.max(val) - np.min(val))
    if np.max(val) <= 1:
        val = (val * 255).astype(np.int32)
    val.clip(0, 255).astype(np.uint8)

    image = PILImage.fromarray(val, mode="RGBA" if val.shape[-1] == 4 else "RGB")
    return image
