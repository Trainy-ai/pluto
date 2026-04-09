import logging
import os
from typing import Any, Dict, Optional, Union

import pluto

from . import sentry as _sentry
from .op import Op
from .sets import Settings, _classify_run_id, _is_display_id, setup
from .util import gen_id, get_char

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'Init'


def _resolve_fork_run_id(fork_run_id: Union[int, str], project: str) -> int:
    """Resolve fork_run_id to a numeric run ID.

    Accepts:
        - int: used as-is
        - numeric string (e.g. "12345"): cast to int
        - display ID (e.g. "T0-123"): resolved via query API
    """
    if isinstance(fork_run_id, int):
        return fork_run_id
    if isinstance(fork_run_id, str) and fork_run_id.isdigit():
        return int(fork_run_id)
    if isinstance(fork_run_id, str) and _is_display_id(fork_run_id):
        import pluto.query as pq

        run = pq.get_run(project, fork_run_id)
        return run['id']
    raise ValueError(
        f'fork_run_id must be a numeric run ID or display ID '
        f'(e.g. "T0-123"), got: {fork_run_id!r}'
    )


class OpInit:
    def __init__(self, config, tags=None, resume=False) -> None:
        self.kwargs = None
        self.config: Dict[str, Any] = config
        self.tags = tags
        self.resume = resume

    def init(self) -> Op:
        op = Op(
            config=self.config,
            settings=self.settings,
            tags=self.tags,
            resume=self.resume,
        )
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
    resume: bool = False,
    fork_run_id: Optional[Union[int, str]] = None,
    fork_step: Optional[int] = None,
    inherit_config: Optional[bool] = None,
    inherit_tags: Optional[bool] = None,
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
        resume: If True, allow resuming an existing run with the same run_id.
                If False (default), raises RuntimeError when a run with the
                same externalId already exists (prevents accidental data
                collision). Runs created via PLUTO_RUN_ID env var are always
                allowed to resume regardless of this flag.
        fork_run_id: Parent run to fork from. Accepts a numeric run ID
            (int), a display ID string (e.g. ``"T0-123"``), or a numeric
            string (e.g. ``"12345"``). Must be used together with fork_step.
        fork_step: Step number to fork at.
            Must be used together with fork_run_id.
        inherit_config: Whether to inherit config from the parent run
            (default: True on server).
        inherit_tags: Whether to inherit tags from the parent run
            (default: False on server).

    Returns:
        Op: The initialized run operation

    Example:
        Single-node training::

            run = pluto.init(project="my-project", name="experiment-1")
            run.log({"loss": 0.5})
            run.finish()

        Fork from an existing run::

            run = pluto.init(
                project="my-project",
                name="lr-tuned",
                fork_run_id="T0-123",   # display ID from the UI
                fork_step=500,
                inherit_config=True,
                config={"lr": 0.01},    # overrides inherited config keys
            )

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
    # Validate fork parameters
    if (fork_run_id is not None) ^ (fork_step is not None):
        raise ValueError('fork_run_id and fork_step must be provided together')

    # TODO: remove legacy compat
    dir = kwargs.get('save_dir', dir)

    settings = setup(settings)
    settings.dir = dir if dir else settings.dir
    settings.project = get_char(project) if project else settings.project
    settings._op_name = (
        get_char(name) if name else gen_id()
    )  # datetime.now().strftime("%Y%m%d"), str(int(time.time()))
    # settings._op_id = id if id else gen_id(seed=settings.project)

    # Classify run_id: display ID → resume, numeric → resume, other → externalId
    # Parameter takes precedence over environment variable (already handled in setup())
    if run_id is not None:
        # Clear any env-var-based classification (explicit param wins)
        settings._resume_run_id = None
        settings._resume_display_id = None
        settings._external_id = None
        _classify_run_id(settings, run_id)

    # Normalize tags before passing to Op
    normalized_tags = [tags] if isinstance(tags, str) else list(tags or [])

    # Auto-add 'konduktor' tag when running inside a Konduktor job
    if os.environ.get('KONDUKTOR_JOB_NAME') and 'konduktor' not in normalized_tags:
        normalized_tags.append('konduktor')

    # Set fork parameters on settings
    if fork_run_id is not None:
        logger.info(
            'Run forking is currently in preview.'
            ' The API may change in future releases.'
        )
        settings._fork_run_id = _resolve_fork_run_id(fork_run_id, settings.project)
        settings._fork_step = fork_step
    if inherit_config is not None:
        settings._inherit_config = inherit_config
    if inherit_tags is not None:
        settings._inherit_tags = inherit_tags

    try:
        op_init = OpInit(config=config, tags=normalized_tags or None, resume=resume)
        op_init.setup(settings=settings)
        op = op_init.init()

        # Set Sentry context for this run
        _sentry.set_tag('project', settings.project)
        _sentry.set_tag('run_id', str(settings._op_id))
        _sentry.set_context(
            'run',
            {
                'project': settings.project,
                'run_id': settings._op_id,
                'run_name': settings._op_name,
                'sync_process': settings.sync_process_enabled,
            },
        )

        return op
    except Exception as e:
        _sentry.capture_exception(e)
        logger.critical('%s: failed, %s', tag, e)  # add early logger
        raise e


def finish(op: Optional[Op] = None) -> None:
    if op:
        op.finish()
    else:
        if pluto.ops:
            for existing_op in pluto.ops:
                existing_op.finish()
