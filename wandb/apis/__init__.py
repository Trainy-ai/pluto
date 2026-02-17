"""Stub for wandb.apis — public API client.

wandb.Api() is used for querying runs, artifacts, etc. This is not
supported by pluto, so we provide a stub that raises informative errors.
"""

import logging

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'WandbCompat.Api'


class Api:
    """Stub for wandb.Api — not supported by pluto compat layer.

    Instantiation succeeds but query methods raise NotImplementedError
    with a helpful message.
    """

    def __init__(self, *args, **kwargs):
        logger.debug('%s: Api instantiated (queries not supported)', tag)

    def _unsupported(self, method):
        raise NotImplementedError(
            f'wandb.Api().{method}() is not supported by the pluto '
            'compatibility layer. Use the pluto API directly.'
        )

    def runs(self, *args, **kwargs):
        self._unsupported('runs')

    def run(self, *args, **kwargs):
        self._unsupported('run')

    def artifact(self, *args, **kwargs):
        self._unsupported('artifact')

    def artifacts(self, *args, **kwargs):
        self._unsupported('artifacts')

    def sweep(self, *args, **kwargs):
        self._unsupported('sweep')


# Also expose at top level since some code does:
# from wandb import Api
# from wandb.apis import Api
__all__ = ['Api']
