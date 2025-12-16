import logging
from typing import Any, Dict, Optional, Union

import mlop

from .op import Op
from .sets import Settings, setup
from .util import gen_id, get_char

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = 'Init'


class OpInit:
    def __init__(self, config) -> None:
        self.kwargs = None
        self.config: Dict[str, Any] = config

    def init(self) -> Op:
        op = Op(config=self.config, settings=self.settings)
        op.settings.meta = []  # TODO: check
        op.start()
        return op

    def setup(self, settings) -> None:
        self.settings = settings


def init(
    dir: Optional[str] = None,
    project: Optional[str] = None,
    name: Optional[str] = None,
    config: Union[dict, str, None] = None,
    settings: Union[Settings, Dict[str, Any], None] = None,
    **kwargs,
) -> Op:
    # TODO: remove legacy compat
    dir = kwargs.get('save_dir', dir)

    settings = setup(settings)
    settings.dir = dir if dir else settings.dir
    settings.project = get_char(project) if project else settings.project
    settings._op_name = (
        get_char(name) if name else gen_id(seed=settings.project)
    )  # datetime.now().strftime("%Y%m%d"), str(int(time.time()))
    # settings._op_id = id if id else gen_id(seed=settings.project)

    try:
        op_init = OpInit(config=config)
        op_init.setup(settings=settings)
        return op_init.init()
    except Exception as e:
        logger.critical('%s: failed, %s', tag, e)  # add early logger
        raise e


def finish(op: Optional[Op] = None) -> None:
    if op:
        op.finish()
    else:
        if mlop.ops:
            for existing_op in mlop.ops:
                existing_op.finish()
