"""Stub for wandb.plot — custom plot helpers.

These are wandb-specific visualization helpers. We provide no-op stubs
so code that calls them doesn't crash.
"""

import logging

from pluto.compat.wandb._coverage import warn_unsupported

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'WandbCompat.Plot'


def line_series(xs, ys, keys=None, title=None, xname=None, **kwargs):
    """No-op stub for wandb.plot.line_series."""
    warn_unsupported("wandb.plot.line_series")
    logger.debug('%s: line_series is not supported', tag)
    return None


def scatter(table, x, y, title=None, **kwargs):
    """No-op stub for wandb.plot.scatter."""
    warn_unsupported("wandb.plot.scatter")
    logger.debug('%s: scatter is not supported', tag)
    return None


def bar(table, label, value, title=None, **kwargs):
    """No-op stub for wandb.plot.bar."""
    warn_unsupported("wandb.plot.bar")
    logger.debug('%s: bar is not supported', tag)
    return None


def histogram(table, value, title=None, **kwargs):
    """No-op stub for wandb.plot.histogram."""
    warn_unsupported("wandb.plot.histogram")
    logger.debug('%s: histogram is not supported', tag)
    return None


def line(table, x, y, stroke=None, title=None, **kwargs):
    """No-op stub for wandb.plot.line."""
    warn_unsupported("wandb.plot.line")
    logger.debug('%s: line is not supported', tag)
    return None


def confusion_matrix(
    y_true=None,
    preds=None,
    class_names=None,
    title=None,
    probs=None,
    **kwargs,
):
    """No-op stub for wandb.plot.confusion_matrix."""
    warn_unsupported("wandb.plot.confusion_matrix")
    logger.debug('%s: confusion_matrix is not supported', tag)
    return None


def roc_curve(
    y_true=None,
    y_probas=None,
    labels=None,
    title=None,
    classes_to_plot=None,
    **kwargs,
):
    """No-op stub for wandb.plot.roc_curve."""
    warn_unsupported("wandb.plot.roc_curve")
    logger.debug('%s: roc_curve is not supported', tag)
    return None


def pr_curve(
    y_true=None,
    y_probas=None,
    labels=None,
    title=None,
    classes_to_plot=None,
    **kwargs,
):
    """No-op stub for wandb.plot.pr_curve."""
    warn_unsupported("wandb.plot.pr_curve")
    logger.debug('%s: pr_curve is not supported', tag)
    return None
