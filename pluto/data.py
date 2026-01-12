import logging

import numpy as np

logger = logging.getLogger(f"{__name__.split('.')[0]}")
tag = 'Data'


class Data:  # TODO: add table class
    tag = tag

    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return {
            **self._data,
            'type': self.__class__.__name__,
            'v': 1,
        }


class Graph(Data):
    tag = 'Graph'

    def __init__(self, data={}):
        self._nodes = data.get('nodes', {})
        self._edges = data.get('edges', {})
        self._successors = data.get('successors', {})
        self._predecessors = data.get('predecessors', {})
        super().__init__(data=self.to_data())

    def add_node(self, node, **attr):
        if node not in self._nodes:
            self._nodes[node] = attr
            self._successors[node] = set()
            self._predecessors[node] = set()
        else:
            self._nodes[node].update(attr)

    def add_edge(self, src, dst, **attr):
        if src not in self._nodes:
            self.add_node(src)
        if dst not in self._nodes:
            self.add_node(dst)

        self._edges[(src, dst)] = attr
        self._successors[src].add(dst)
        self._predecessors[dst].add(src)

    @property
    def nodes(self):
        return self._nodes

    def edges(self, data=False):
        if data:
            return list(self._edges.items())
        return list(self._edges.keys())

    def to_data(self):
        return {
            'nodes': self._nodes,
            'edges': self._edges,
            'successors': self._successors,
            'predecessors': self._predecessors,
        }


class Histogram(Data):
    tag = 'Histogram'

    def __init__(self, data, bins=64):
        self._shape = 'generic'
        if not isinstance(bins, int) and len(data) == 2:
            # TODO: support non-uniform bins
            logger.debug(
                f'{tag}: using pre-set bins from data; '
                'bins need to have uniform intervals'
            )
            d = data[0].tolist() if hasattr(data[0], 'tolist') else data[0]
            b = data[1].tolist() if hasattr(data[1], 'tolist') else data[1]
            if len(d) + 1 != len(b):
                logger.critical(
                    f'{self.tag}: data and length must be the same length: '
                    'force proceeding'
                )
            else:
                self._shape = 'uniform'
        elif (
            isinstance(data, list)
            or isinstance(data, np.ndarray)
            or (data.__class__.__name__ == 'Tensor' and data.ndim == 1)
        ):
            d, b = np.histogram(data, bins=bins)
            d, b = d.tolist(), b.tolist()
            self._shape = 'uniform'
        else:
            logger.critical(
                f'{self.tag}: data must be a list or numpy array: '
                'force proceeding with an empty histogram'
            )
            d, b = [0], [0, 1]

        self._freq = d
        self._bins = b
        super().__init__(data=self.to_data())

    def to_data(self):
        if self._shape == 'uniform':
            return {
                'freq': self._freq,
                'maxFreq': max(self._freq),
                'bins': {
                    'min': min(self._bins),
                    'max': max(self._bins),
                    'num': len(self._bins) - 1,
                },
                'shape': self._shape,
            }
        else:
            return {
                'freq': self._freq,
                'bins': self._bins,
                'shape': self._shape,
            }


class Table(Data):
    tag = 'Table'

    def __init__(self, data=None, **kwargs):
        # TODO: remove legacy compat
        self._col = kwargs.get('columns', [])
        self._row = kwargs.get('rows', [])
        self._table = kwargs.get('table', [[None]])
        self._dataframe = kwargs.get('dataframe', None)
        data = self._dataframe if self._dataframe is not None else data

        if data.__class__.__name__ == 'DataFrame' and hasattr(data, 'to_dict'):
            self._pd = self._dataframe.to_dict(orient='split')
            self._table = self._pd.get('data', [[None]])
            if not (self._col and len(self._col) == len(self._table[0])):
                self._col = self._pd.get('columns', [])
            if not (self._row and len(self._row) == len(self._table)):
                self._row = self._pd.get('index', [])
        elif isinstance(data, np.ndarray):
            self._table = data.tolist()
        elif isinstance(data, list):
            self._table = data
        else:
            logger.warning(
                f'{self.tag}: unsupported data type: {data.__class__.__name__}'
            )

        # TODO: remove dtype enforcement
        for i, r in enumerate(self._table):
            for j, c in enumerate(r):
                if self._col and not isinstance(c, type(self._table[0][j])):
                    logger.warning(
                        f'{self.tag}: type mismatch detected in column '
                        f'{j}: force proceeding'
                    )
                    c = self._table[0][j]
                if self._row and not isinstance(c, type(self._table[i][0])):
                    logger.warning(
                        f'{self.tag}: type mismatch detected in row '
                        f'{i}: force proceeding'
                    )
                    c = self._table[i][0]

        super().__init__(data=self.to_data())

    def to_data(self):
        return {
            'table': self._table,
            **(
                {
                    'col': [
                        {'name': str(c), 'dtype': self._table[0][i].__class__.__name__}
                        for i, c in enumerate(self._col)
                    ]
                }
                if len(self._col) == len(self._table[0])
                else {}
            ),
            **(
                {
                    'row': [
                        {'name': str(r), 'dtype': self._table[i][0].__class__.__name__}
                        for i, r in enumerate(self._row)
                    ]
                }
                if len(self._row) == len(self._table)
                else {}
            ),
        }
