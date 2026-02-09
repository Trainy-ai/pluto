import logging
import os
from typing import Any, Dict, Optional, Union

import pluto

from .op import Op
from .sets import Settings, setup
from .util import gen_id, get_char

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'Init'


class OpInit:
    def __init__(self, config, tags=None) -> None:
        self.kwargs = None
        self.config: Dict[str, Any] = config
        self.tags = tags

    def init(self) -> Op:
        op = Op(config=self.config, settings=self.settings, tags=self.tags)
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
    tags: Union[str, list[str], None] = None,
    run_id: Optional[str] = None,
    **kwargs,
) -> Op:
    """
    Initialize a new Pluto run.

    Args:
        dir: Directory for storing run artifacts
        project: Project name
        name: Run name. For multi-node training with run_id, use the same name
              across all ranks - the name is only used when creating a new run
              and is ignored when resuming an existing run.
        config: Run configuration dict
        settings: Settings object or dict
        tags: Single tag or list of tags
        run_id: User-provided run ID for multi-node distributed training.
                When multiple processes use the same run_id, they will all
                log to the same run (Neptune-style resume). Can also be set
                via PLUTO_RUN_ID environment variable.

    Returns:
        Op: The initialized run operation

    Example:
        Single-node training::

            run = pluto.init(project="my-project", name="experiment-1")
            run.log({"loss": 0.5})
            run.finish()

        Multi-node distributed training::

            # Set shared run_id before launching (e.g., in launch script)
            # export PLUTO_RUN_ID="ddp-experiment-$(date +%Y%m%d)"

            # In training script - all ranks use the same name
            run = pluto.init(
                project="my-project",
                name="ddp-training",  # Use same name for all ranks
                run_id=os.environ.get("PLUTO_RUN_ID"),
            )

            # Check if this rank resumed an existing run
            if run.resumed:
                print(f"Resumed run {run.id}")

            # Log with rank-prefixed metrics
            run.log({f"loss/rank{rank}": loss_value})

    Note:
        When using ``run_id`` for multi-node training, the ``name`` parameter
        is only used by the first process that creates the run. Subsequent
        processes that resume the run will use the original name. For clarity,
        use the same ``name`` value across all ranks.
    """
    # TODO: remove legacy compat
    dir = kwargs.get('save_dir', dir)

    settings = setup(settings)
    settings.dir = dir if dir else settings.dir
    settings.project = get_char(project) if project else settings.project
    settings._op_name = (
        get_char(name) if name else gen_id(seed=settings.project)
    )  # datetime.now().strftime("%Y%m%d"), str(int(time.time()))
    # settings._op_id = id if id else gen_id(seed=settings.project)

    # Set external_id for multi-node distributed training
    # Parameter takes precedence over environment variable (already handled in setup())
    if run_id is not None:
        settings._external_id = run_id

    # Auto-add 'konduktor' tag when running inside a Konduktor job
    if os.environ.get('KONDUKTOR_JOB_NAME'):
        if tags is None:
            tags = ['konduktor']
        elif isinstance(tags, str):
            tags = [tags, 'konduktor'] if tags != 'konduktor' else [tags]
        elif 'konduktor' not in tags:
            tags = list(tags) + ['konduktor']

    # Normalize tags before passing to Op
    normalized_tags = None
    if tags:
        if isinstance(tags, str):
            normalized_tags = [tags]
        else:
            normalized_tags = list(tags)

    try:
        op_init = OpInit(config=config, tags=normalized_tags)
        op_init.setup(settings=settings)
        op = op_init.init()

        return op
    except Exception as e:
        logger.critical('%s: failed, %s', tag, e)  # add early logger
        raise e


def finish(op: Optional[Op] = None) -> None:
    if op:
        op.finish()
    else:
        if pluto.ops:
            for existing_op in pluto.ops:
                existing_op.finish()
