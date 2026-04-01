import contextlib
import logging
from typing import Dict, List, Optional, Union

import pluto
from pluto.util import import_lib

optuna = import_lib('optuna')

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'Optuna'


def _is_multi_objective(study) -> bool:
    return len(study.directions) > 1


def _is_pruned(trial) -> bool:
    return trial.state == optuna.trial.TrialState.PRUNED


def _get_objective_names(
    study, target_names: Optional[List[str]] = None
) -> Union[List[str], str]:
    if _is_multi_objective(study):
        if target_names is not None:
            assert len(target_names) == len(study.directions), (
                f'target_names length ({len(target_names)}) must match '
                f'study.directions length ({len(study.directions)})'
            )
            return target_names
        return [f'objective_{i}' for i in range(len(study.directions))]
    else:
        if target_names is not None:
            assert len(target_names) == 1, (
                f'target_names length ({len(target_names)}) must be 1 '
                f'for single-objective study'
            )
            return target_names[0]
        return 'objective_value'


class PlutoCallback:
    """Optuna callback that logs study progress to Pluto.

    Logs trial parameters, values, best trials, study metadata, and
    optionally Optuna visualization plots to a Pluto run.

    Each trial's parameters and objective values are logged as metrics so
    they appear on the Pluto dashboard. Study-level metadata (directions,
    best values, distributions) is stored in the run config. Optuna
    visualization plots are logged as HTML images at a configurable
    frequency.

    When ``create_run=True`` (the default), the callback creates a
    dedicated Pluto run for the study. The run URL is printed to the
    console so you can follow progress live. Pass an existing ``Op``
    via the ``op`` parameter to log into an already-active run instead.

    Args:
        op: Existing Pluto ``Op`` to log to. When ``None``, uses the
            last active run or creates a new one (controlled by
            ``create_run``).
        project: Project name for the auto-created run.
        name: Run name for the auto-created run.
        tags: Tags for the auto-created run.
        create_run: If ``True`` (default) and no ``op`` is provided and
            no run is already active, create a new Pluto run.
        target_names: Human-readable names for objectives. Length must
            match the number of study directions. Defaults to
            ``objective_value`` (single) or ``objective_0``, ... (multi).
        plots_update_freq: Log plots every N completed trials, or
            ``"never"`` to disable. Default ``10``.
        log_plot_contour: Log contour plot. Default ``True``.
        log_plot_edf: Log EDF plot. Default ``True``.
        log_plot_parallel_coordinate: Log parallel coordinate plot.
            Default ``True``.
        log_plot_param_importances: Log parameter importances plot.
            Default ``True``.
        log_plot_pareto_front: Log Pareto front plot (multi-objective
            only). Default ``True``.
        log_plot_slice: Log slice plot. Default ``True``.
        log_plot_intermediate_values: Log intermediate values plot.
            Default ``True``.
        log_plot_optimization_history: Log optimization history plot.
            Default ``True``.
        visualization_backend: ``"plotly"`` (default) or ``"matplotlib"``.
        log_study_to_notes: If ``True`` (default), write a Markdown
            summary to the run notes after every trial, including
            links to per-trial Pluto runs when ``trial_run_urls`` are
            provided.

    Example::

        import optuna
        import pluto
        from pluto.compat.optuna import PlutoCallback

        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return (x - 2) ** 2

        callback = PlutoCallback(project="hpo", name="optuna-sweep")
        study = optuna.create_study()
        study.optimize(objective, n_trials=100, callbacks=[callback])
    """

    def __init__(
        self,
        op=None,
        *,
        project: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[Union[str, List[str]]] = None,
        create_run: bool = True,
        target_names: Optional[List[str]] = None,
        plots_update_freq: Union[int, str] = 10,
        log_plot_contour: bool = True,
        log_plot_edf: bool = True,
        log_plot_parallel_coordinate: bool = True,
        log_plot_param_importances: bool = True,
        log_plot_pareto_front: bool = True,
        log_plot_slice: bool = True,
        log_plot_intermediate_values: bool = True,
        log_plot_optimization_history: bool = True,
        visualization_backend: str = 'plotly',
        log_study_to_notes: bool = True,
    ):
        self._op = op
        self._project = project
        self._name = name
        self._tags = tags
        self._create_run = create_run
        self._target_names = target_names
        self._objective_names: Optional[Union[List[str], str]] = None

        self._plots_update_freq = plots_update_freq
        self._log_plot_contour = log_plot_contour
        self._log_plot_edf = log_plot_edf
        self._log_plot_parallel_coordinate = log_plot_parallel_coordinate
        self._log_plot_param_importances = log_plot_param_importances
        self._log_plot_pareto_front = log_plot_pareto_front
        self._log_plot_slice = log_plot_slice
        self._log_plot_intermediate_values = log_plot_intermediate_values
        self._log_plot_optimization_history = log_plot_optimization_history
        self._visualization_backend = visualization_backend
        self._log_study_to_notes = log_study_to_notes

        # Mapping from Optuna trial number → Pluto run URL
        # Users can populate this to get clickable links in notes
        self.trial_run_urls: Dict[int, str] = {}

        self._initialized = False

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def op(self):
        """The Pluto ``Op`` being logged to (may be ``None`` before first call)."""
        return self._op

    def set_trial_run_url(self, trial_number: int, url: str) -> None:
        """Register the Pluto run URL for a specific trial.

        When ``log_study_to_notes=True``, registered URLs appear as
        clickable Markdown links in the study notes.

        Args:
            trial_number: The Optuna ``trial.number``.
            url: The Pluto run URL (e.g. ``op.settings.url_view``).

        Example::

            def objective(trial):
                run = pluto.init(project="hpo", name=f"trial-{trial.number}")
                # ... training ...
                callback.set_trial_run_url(trial.number, run.settings.url_view)
                run.finish()
                return loss

            callback = PlutoCallback(project="hpo", name="optuna-study")
            study.optimize(objective, n_trials=50, callbacks=[callback])
        """
        self.trial_run_urls[trial_number] = url

    # ------------------------------------------------------------------
    # Optuna callback interface
    # ------------------------------------------------------------------

    def __call__(self, study, trial) -> None:
        if not self._initialized:
            self._init_run(study)

        self._log_trial(study, trial)
        self._log_best_trials(study)
        self._log_study_details(study, trial)
        self._log_plots(study, trial)
        if self._log_study_to_notes:
            self._update_notes(study)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_run(self, study) -> None:
        self._objective_names = _get_objective_names(study, self._target_names)

        if self._op is None:
            if pluto.ops and len(pluto.ops) > 0:
                self._op = pluto.ops[-1]
            elif self._create_run:
                config = {
                    'study_name': study.study_name,
                    'directions': [str(d) for d in study.directions]
                    if _is_multi_objective(study)
                    else str(study.direction),
                }
                self._op = pluto.init(
                    project=self._project or 'optuna',
                    name=self._name,
                    tags=self._tags,
                    config=config,
                )

        if self._op is not None:
            self._op.add_tags('optuna')

        self._initialized = True

    # ------------------------------------------------------------------
    # Per-trial logging
    # ------------------------------------------------------------------

    def _log_trial(self, study, trial) -> None:
        if self._op is None:
            return

        obj_names = self._objective_names or 'objective_value'
        data: Dict[str, object] = {}

        # Log trial params as metrics
        for param_name, param_value in trial.params.items():
            if isinstance(param_value, (int, float)):
                data[f'trial/param/{param_name}'] = param_value

        # Log objective value(s)
        if not _is_pruned(trial) and trial.values is not None:
            if _is_multi_objective(study):
                for i, v in enumerate(trial.values):
                    name = obj_names[i] if isinstance(obj_names, list) else obj_names
                    data[f'trial/{name}'] = v
            elif trial.value is not None:
                name = obj_names if isinstance(obj_names, str) else obj_names[0]
                data[f'trial/{name}'] = trial.value

        # Log trial metadata
        data['trial/number'] = trial.number
        data['trial/state'] = repr(trial.state)
        if trial.duration is not None:
            data['trial/duration_s'] = trial.duration.total_seconds()

        self._op.log(data, step=trial.number)

    # ------------------------------------------------------------------
    # Best trials
    # ------------------------------------------------------------------

    def _log_best_trials(self, study) -> None:
        if self._op is None:
            return

        completed = [
            t for t in study.get_trials() if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            return

        obj_names = self._objective_names or 'objective_value'
        data: Dict[str, object] = {}

        if _is_multi_objective(study):
            for i, bt in enumerate(study.best_trials):
                for j, v in enumerate(bt.values):
                    name = obj_names[j] if isinstance(obj_names, list) else obj_names
                    data[f'best/pareto_{i}/{name}'] = v
        else:
            best = study.best_trial
            name = obj_names if isinstance(obj_names, str) else obj_names[0]
            data[f'best/{name}'] = best.value
            data['best/trial_number'] = best.number
            for pname, pval in best.params.items():
                if isinstance(pval, (int, float)):
                    data[f'best/param/{pname}'] = pval

        self._op.log(data, step=len(completed))

    # ------------------------------------------------------------------
    # Study details (logged once on first trial)
    # ------------------------------------------------------------------

    def _log_study_details(self, study, trial) -> None:
        if self._op is None:
            return

        if trial.number != 0:
            return

        config: Dict[str, object] = {
            'study_name': study.study_name,
        }
        if _is_multi_objective(study):
            config['directions'] = [str(d) for d in study.directions]
        else:
            config['direction'] = str(study.direction)

        if study.user_attrs:
            config['user_attrs'] = study.user_attrs

        self._op.config.update(config)

    # ------------------------------------------------------------------
    # Visualization plots
    # ------------------------------------------------------------------

    def _should_log_plots(self, study, trial) -> bool:
        if self._plots_update_freq == 'never':
            return False
        completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
        if not completed:
            return False
        return trial.number % self._plots_update_freq == 0

    def _log_plots(self, study, trial) -> None:
        if self._op is None:
            return
        if not self._should_log_plots(study, trial):
            return

        if self._visualization_backend == 'matplotlib':
            try:
                import optuna.visualization.matplotlib as vis
            except ImportError:
                logger.warning('%s: matplotlib visualization not available', tag)
                return
        elif self._visualization_backend == 'plotly':
            try:
                import optuna.visualization as vis
            except ImportError:
                logger.warning('%s: plotly visualization not available', tag)
                return
        else:
            logger.error(
                '%s: unsupported visualization backend: %s',
                tag,
                self._visualization_backend,
            )
            return

        if not getattr(vis, 'is_available', lambda: True)():
            return

        directions = study.directions
        obj_names = self._objective_names or 'objective_value'

        for i in range(len(directions)):
            if isinstance(obj_names, list):
                target = lambda t, idx=i: t.values[idx]  # noqa: E731
                target_name: str = obj_names[i]
                prefix = f'plots/{obj_names[i]}'
            else:
                target = None
                target_name = obj_names
                prefix = 'plots'

            params = [p for t in study.trials for p in t.params.keys()]

            if self._log_plot_contour and params:
                with contextlib.suppress(Exception):
                    fig = vis.plot_contour(
                        study, target=target, target_name=target_name
                    )
                    self._op.log(
                        {f'{prefix}/contour': _figure_to_image(fig)},
                        step=trial.number,
                    )

            if self._log_plot_edf:
                with contextlib.suppress(Exception):
                    fig = vis.plot_edf(study, target=target, target_name=target_name)
                    self._op.log(
                        {f'{prefix}/edf': _figure_to_image(fig)},
                        step=trial.number,
                    )

            if self._log_plot_parallel_coordinate:
                with contextlib.suppress(Exception):
                    fig = vis.plot_parallel_coordinate(
                        study, target=target, target_name=target_name
                    )
                    self._op.log(
                        {f'{prefix}/parallel_coordinate': _figure_to_image(fig)},
                        step=trial.number,
                    )

            if self._log_plot_param_importances:
                trials_for_importance = study.get_trials(
                    states=(
                        optuna.trial.TrialState.COMPLETE,
                        optuna.trial.TrialState.PRUNED,
                    )
                )
                if len(trials_for_importance) > 1:
                    with contextlib.suppress(
                        RuntimeError, ValueError, ZeroDivisionError, Exception
                    ):
                        fig = vis.plot_param_importances(
                            study, target=target, target_name=target_name
                        )
                        self._op.log(
                            {f'{prefix}/param_importances': _figure_to_image(fig)},
                            step=trial.number,
                        )

            if self._log_plot_slice and params:
                with contextlib.suppress(Exception):
                    fig = vis.plot_slice(study, target=target, target_name=target_name)
                    self._op.log(
                        {f'{prefix}/slice': _figure_to_image(fig)},
                        step=trial.number,
                    )

            if (
                not _is_multi_objective(study)
                and self._log_plot_intermediate_values
                and any(t.intermediate_values for t in study.trials)
            ):
                with contextlib.suppress(Exception):
                    fig = vis.plot_intermediate_values(study)
                    self._op.log(
                        {f'{prefix}/intermediate_values': _figure_to_image(fig)},
                        step=trial.number,
                    )

            if self._log_plot_optimization_history:
                with contextlib.suppress(Exception):
                    fig = vis.plot_optimization_history(
                        study, target=target, target_name=target_name
                    )
                    self._op.log(
                        {f'{prefix}/optimization_history': _figure_to_image(fig)},
                        step=trial.number,
                    )

        if (
            self._log_plot_pareto_front
            and _is_multi_objective(study)
            and self._visualization_backend == 'plotly'
        ):
            with contextlib.suppress(Exception):
                fig = vis.plot_pareto_front(study, target_names=obj_names)
                self._op.log(
                    {'plots/pareto_front': _figure_to_image(fig)},
                    step=trial.number,
                )

    # ------------------------------------------------------------------
    # Notes (Markdown summary with links to trial runs)
    # ------------------------------------------------------------------

    def _update_notes(self, study) -> None:
        if self._op is None:
            return

        completed = [
            t for t in study.get_trials() if t.state == optuna.trial.TrialState.COMPLETE
        ]
        pruned = [
            t for t in study.get_trials() if t.state == optuna.trial.TrialState.PRUNED
        ]
        total = len(study.trials)

        lines: List[str] = []
        lines.append(f'# Optuna Study: {study.study_name}')
        lines.append('')

        if _is_multi_objective(study):
            lines.append(
                f'**Directions:** {", ".join(str(d) for d in study.directions)}'
            )
        else:
            lines.append(f'**Direction:** {study.direction}')

        lines.append(
            f'**Trials:** {total} total, '
            f'{len(completed)} completed, '
            f'{len(pruned)} pruned'
        )
        lines.append('')

        # Best trial(s)
        if completed:
            if _is_multi_objective(study):
                lines.append(f'## Pareto Front ({len(study.best_trials)} trials)')
                lines.append('')
                for bt in study.best_trials:
                    vals = ', '.join(f'{v:.6g}' for v in bt.values)
                    url = self.trial_run_urls.get(bt.number)
                    trial_ref = (
                        f'[Trial {bt.number}]({url})' if url else f'Trial {bt.number}'
                    )
                    lines.append(f'- {trial_ref}: [{vals}]')
                lines.append('')
            else:
                best = study.best_trial
                url = self.trial_run_urls.get(best.number)
                trial_ref = (
                    f'[Trial {best.number}]({url})' if url else f'Trial {best.number}'
                )
                lines.append('## Best Trial')
                lines.append('')
                lines.append(f'- **{trial_ref}** — value: `{best.value:.6g}`')
                params_str = ', '.join(
                    f'`{k}`={v:.6g}' if isinstance(v, float) else f'`{k}`={v}'
                    for k, v in best.params.items()
                )
                if params_str:
                    lines.append(f'- Params: {params_str}')
                lines.append('')

        # Trial table
        all_trials = study.trials
        if all_trials:
            lines.append('## Trials')
            lines.append('')

            # Build header
            param_names = sorted({p for t in all_trials for p in t.params.keys()})
            obj_names_raw = self._objective_names or 'objective_value'
            obj_names: List[str] = (
                [obj_names_raw]
                if isinstance(obj_names_raw, str)
                else list(obj_names_raw)
            )

            header_parts = ['Trial', 'State'] + obj_names + param_names
            if self.trial_run_urls:
                header_parts.append('Run')
            lines.append('| ' + ' | '.join(header_parts) + ' |')
            lines.append('| ' + ' | '.join(['---'] * len(header_parts)) + ' |')

            for t in all_trials:
                state = 'PRUNED' if _is_pruned(t) else t.state.name
                row = [str(t.number), state]

                # Objective values
                if t.values is not None and not _is_pruned(t):
                    row.extend(f'{v:.6g}' for v in t.values)
                elif t.value is not None and not _is_pruned(t):
                    row.append(f'{t.value:.6g}')
                else:
                    row.extend(['—'] * len(obj_names))

                # Params
                for p in param_names:
                    val = t.params.get(p)
                    if val is None:
                        row.append('—')
                    elif isinstance(val, float):
                        row.append(f'{val:.6g}')
                    else:
                        row.append(str(val))

                # Run link
                if self.trial_run_urls:
                    url = self.trial_run_urls.get(t.number)
                    row.append(f'[link]({url})' if url else '—')

                lines.append('| ' + ' | '.join(row) + ' |')
            lines.append('')

        notes_md = '\n'.join(lines)
        self._op.log({'notes': pluto.Text(notes_md)})


# ------------------------------------------------------------------
# Standalone helper: log a completed study after the fact
# ------------------------------------------------------------------


def log_study_metadata(
    study,
    op=None,
    *,
    project: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[Union[str, List[str]]] = None,
    target_names: Optional[List[str]] = None,
    log_plots: bool = True,
    log_trials: bool = True,
    visualization_backend: str = 'plotly',
    log_plot_contour: bool = True,
    log_plot_edf: bool = True,
    log_plot_parallel_coordinate: bool = True,
    log_plot_param_importances: bool = True,
    log_plot_pareto_front: bool = True,
    log_plot_slice: bool = True,
    log_plot_intermediate_values: bool = True,
    log_plot_optimization_history: bool = True,
    trial_run_urls: Optional[Dict[int, str]] = None,
) -> None:
    """Log a completed Optuna study to a Pluto run.

    This is a convenience function for logging study results after
    ``study.optimize()`` has finished. For live logging during
    optimization, use :class:`PlutoCallback` instead.

    Args:
        study: The completed Optuna study.
        op: Existing Pluto ``Op``. If ``None``, the last active run is
            used or a new run is created.
        project: Project name (used when creating a new run).
        name: Run name (used when creating a new run).
        tags: Tags (used when creating a new run).
        target_names: Human-readable objective names.
        log_plots: Whether to generate and log visualization plots.
        log_trials: Whether to log per-trial data.
        visualization_backend: ``"plotly"`` or ``"matplotlib"``.
        trial_run_urls: Mapping from trial number to Pluto run URL.
            When provided, the notes Markdown table includes clickable
            links to the individual trial runs.
        log_plot_*: Toggle individual plot types.

    Example::

        study = optuna.create_study()
        study.optimize(objective, n_trials=100)

        from pluto.compat.optuna import log_study_metadata
        log_study_metadata(study, project="hpo", name="study-results")
    """
    callback = PlutoCallback(
        op=op,
        project=project,
        name=name,
        tags=tags,
        target_names=target_names,
        plots_update_freq=1 if log_plots else 'never',
        log_plot_contour=log_plot_contour,
        log_plot_edf=log_plot_edf,
        log_plot_parallel_coordinate=log_plot_parallel_coordinate,
        log_plot_param_importances=log_plot_param_importances,
        log_plot_pareto_front=log_plot_pareto_front,
        log_plot_slice=log_plot_slice,
        log_plot_intermediate_values=log_plot_intermediate_values,
        log_plot_optimization_history=log_plot_optimization_history,
        visualization_backend=visualization_backend,
    )

    if trial_run_urls:
        callback.trial_run_urls = trial_run_urls

    callback._init_run(study)

    # Log config
    callback._log_study_details(study, study.trials[0] if study.trials else None)

    # Log each trial
    if log_trials:
        for trial in study.trials:
            callback._log_trial(study, trial)

    # Log best
    callback._log_best_trials(study)

    # Plots (use last trial as trigger)
    if log_plots and study.trials:
        last_trial = study.trials[-1]
        callback._log_plots(study, last_trial)

    # Notes summary
    if callback._log_study_to_notes:
        callback._update_notes(study)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _figure_to_image(fig):
    """Convert a plotly or matplotlib figure to a pluto.Image."""
    import io

    # Plotly figure
    if hasattr(fig, 'to_image'):
        img_bytes = fig.to_image(format='png')
        return pluto.Image(data=img_bytes, ext='.png')

    # Matplotlib figure
    if hasattr(fig, 'savefig'):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        import matplotlib.pyplot as plt

        plt.close(fig)
        return pluto.Image(data=buf.read(), ext='.png')

    # Matplotlib Axes (get parent figure)
    if hasattr(fig, 'get_figure'):
        return _figure_to_image(fig.get_figure())

    logger.warning('%s: cannot convert figure of type %s', tag, type(fig).__name__)
    return pluto.Text(str(fig))
