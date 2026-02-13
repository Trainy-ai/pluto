"""wandb-compatible data type wrappers that convert to pluto equivalents.

Each class accepts wandb-style constructor arguments and provides a
_to_pluto() method that returns the corresponding pluto type.
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'WandbCompat.DataTypes'


class Image:
    """wandb.Image-compatible wrapper.

    Accepts the same constructor args as wandb.Image and converts to
    pluto.file.Image via _to_pluto().
    """

    def __init__(
        self,
        data_or_path: Any = None,
        mode: Optional[str] = None,
        caption: Optional[str] = None,
        grouping: Optional[int] = None,
        classes: Any = None,
        boxes: Any = None,
        masks: Any = None,
        file_type: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        self.data_or_path = data_or_path
        self.caption = caption
        self._mode = mode

    def _to_pluto(self) -> Any:
        from pluto.file import Image as PlutoImage

        return PlutoImage(data=self.data_or_path, caption=self.caption)


class Audio:
    """wandb.Audio-compatible wrapper."""

    def __init__(
        self,
        data_or_path: Any = None,
        sample_rate: Optional[int] = None,
        caption: Optional[str] = None,
    ) -> None:
        self.data_or_path = data_or_path
        self.sample_rate = sample_rate
        self.caption = caption

    def _to_pluto(self) -> Any:
        from pluto.file import Audio as PlutoAudio

        return PlutoAudio(
            data=self.data_or_path,
            sample_rate=self.sample_rate,
            caption=self.caption,
        )


class Video:
    """wandb.Video-compatible wrapper."""

    def __init__(
        self,
        data_or_path: Any = None,
        caption: Optional[str] = None,
        fps: Optional[int] = None,
        format: Optional[str] = None,
    ) -> None:
        self.data_or_path = data_or_path
        self.caption = caption
        self.fps = fps
        self.format = format

    def _to_pluto(self) -> Any:
        from pluto.file import Video as PlutoVideo

        return PlutoVideo(
            data=self.data_or_path,
            fps=self.fps,
            caption=self.caption,
            format=self.format,
        )


class Table:
    """wandb.Table-compatible wrapper."""

    MAX_ROWS = 10_000
    MAX_ARTIFACT_ROWS = 200_000

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        data: Optional[List[List[Any]]] = None,
        rows: Optional[List[List[Any]]] = None,
        dataframe: Any = None,
        dtype: Any = None,
        optional: Any = True,
        allow_mixed_types: bool = False,
    ) -> None:
        self.columns = columns or []
        self._data: List[List[Any]] = data or rows or []
        self._dataframe = dataframe

    def add_data(self, *row: Any) -> None:
        self._data.append(list(row))

    def add_column(self, name: str, data: List[Any]) -> None:
        self.columns.append(name)
        for i, val in enumerate(data):
            if i < len(self._data):
                self._data[i].append(val)
            else:
                self._data.append([val])

    def get_column(self, name: str, convert_to: Optional[str] = None) -> List[Any]:
        if name in self.columns:
            idx = self.columns.index(name)
            return [row[idx] for row in self._data if idx < len(row)]
        return []

    def _to_pluto(self) -> Any:
        from pluto.data import Table as PlutoTable

        if self._dataframe is not None:
            return PlutoTable(
                data=None,
                dataframe=self._dataframe,
                columns=self.columns or None,
            )
        return PlutoTable(
            data=self._data,
            columns=self.columns or None,
        )


class Histogram:
    """wandb.Histogram-compatible wrapper."""

    def __init__(
        self,
        sequence: Optional[Any] = None,
        np_histogram: Optional[Any] = None,
        num_bins: int = 64,
    ) -> None:
        self.sequence = sequence
        self.np_histogram = np_histogram
        self.num_bins = num_bins

    def _to_pluto(self) -> Any:
        from pluto.data import Histogram as PlutoHistogram

        if self.np_histogram is not None:
            # np_histogram is a tuple of (values, bin_edges)
            return PlutoHistogram(data=self.np_histogram, bins=self.np_histogram[1])
        if self.sequence is not None:
            return PlutoHistogram(data=self.sequence, bins=self.num_bins)
        return PlutoHistogram(data=[0], bins=1)


class Html:
    """wandb.Html-compatible wrapper. Maps to pluto.file.Text."""

    def __init__(
        self,
        data: Any = None,
        inject: bool = True,
    ) -> None:
        if hasattr(data, 'read'):
            self._html = data.read()
        elif isinstance(data, str) and os.path.isfile(data):
            with open(data) as f:
                self._html = f.read()
        else:
            self._html = str(data) if data is not None else ''

    def _to_pluto(self) -> Any:
        from pluto.file import Text as PlutoText

        return PlutoText(data=self._html)


class Graph:
    """wandb.Graph-compatible wrapper."""

    def __init__(self) -> None:
        self._nodes: List[Any] = []
        self._edges: List[Any] = []

    def _to_pluto(self) -> Any:
        from pluto.data import Graph as PlutoGraph

        data: Dict[str, Any] = {'nodes': {}, 'edges': {}}
        for i, node in enumerate(self._nodes):
            name = getattr(node, 'name', str(i))
            data['nodes'][name] = {}
        for edge in self._edges:
            src = edge[0] if isinstance(edge, (list, tuple)) else str(edge)
            dst = edge[1] if isinstance(edge, (list, tuple)) else str(edge)
            data['edges'][(src, dst)] = {}
        return PlutoGraph(data=data)


class Artifact:
    """wandb.Artifact-compatible wrapper for file collections.

    wandb's Artifact is a versioned collection of files. Pluto's Artifact is
    a single file. This wrapper collects files and logs each individually
    when log_artifact() is called.
    """

    def __init__(
        self,
        name: str,
        type: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        incremental: bool = False,
        use_as: Optional[str] = None,
    ) -> None:
        self.name = name
        self.type = type
        self.description = description
        self.metadata = metadata or {}
        self._files: List[Dict[str, Any]] = []

    def add_file(
        self,
        local_path: str,
        name: Optional[str] = None,
        is_tmp: bool = False,
        skip_cache: bool = False,
        policy: str = 'mutable',
    ) -> 'Artifact':
        self._files.append(
            {
                'path': local_path,
                'name': name or os.path.basename(local_path),
            }
        )
        return self

    def add_dir(
        self,
        local_path: str,
        name: Optional[str] = None,
        skip_cache: bool = False,
        policy: str = 'mutable',
    ) -> 'Artifact':
        for root, _dirs, files in os.walk(local_path):
            for f in files:
                fpath = os.path.join(root, f)
                rel = os.path.relpath(fpath, local_path)
                if name:
                    rel = os.path.join(name, rel)
                self._files.append({'path': fpath, 'name': rel})
        return self

    def add_reference(
        self,
        uri: str,
        name: Optional[str] = None,
        checksum: bool = True,
        max_objects: Optional[int] = None,
    ) -> 'Artifact':
        logger.debug('%s: add_reference is not supported, ignoring', tag)
        return self

    def _to_pluto_files(self) -> List[Any]:
        """Convert collected files to pluto Artifact objects."""
        from pluto.file import Artifact as PlutoArtifact

        result = []
        for entry in self._files:
            result.append(
                PlutoArtifact(
                    data=entry['path'],
                    caption=entry['name'],
                    metadata=self.metadata,
                )
            )
        return result

    # Stubs for download/verify (not supported)
    def download(self, root: Optional[str] = None, **kwargs: Any) -> str:
        logger.debug('%s: download is not supported', tag)
        return root or '.'

    def verify(self, root: Optional[str] = None) -> bool:
        logger.debug('%s: verify is not supported', tag)
        return True

    def new_draft(self) -> 'Artifact':
        return self


class AlertLevel:
    """wandb.AlertLevel-compatible enum."""

    INFO = 'INFO'
    WARN = 'WARN'
    ERROR = 'ERROR'
