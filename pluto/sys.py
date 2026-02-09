import importlib.metadata
import logging
import os
import platform
import time
from typing import Any, Dict, List, Mapping, Optional, Union, cast

import psutil

from .sets import Settings
from .util import run_cmd, to_human  # TODO: move to server side

logger = logging.getLogger(f'{__name__.split(".")[0]}')
tag = 'System'


class System:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        # State tracking for rate calculations
        self._prev_counters: Dict[str, float] = {}
        self._prev_timestamp: Optional[float] = None

        self.uname: Dict[str, str] = platform.uname()._asdict()
        self.timezone: List[str] = list(time.tzname)

        self.cpu_count = psutil.cpu_count
        try:  # perf
            self.cpu_freq: List[Dict[str, float]] = [
                i._asdict() for i in psutil.cpu_freq(percpu=True)
            ]
        except Exception:  # errors on darwin t81xx
            self.cpu_freq = [{'current': 0, 'min': 0, 'max': 0}]

        self.svmem: Dict[str, Any] = psutil.virtual_memory()._asdict()
        self.sswap: Dict[str, Any] = psutil.swap_memory()._asdict()
        self.disk: List[Dict[str, Any]] = [
            i._asdict() for i in psutil.disk_partitions()
        ]
        self.net_if_addrs: Dict[str, List[Dict[str, Any]]] = {
            i: [
                {k: v for k, v in j._asdict().items() if k != 'family'}
                for j in psutil.net_if_addrs()[i]
            ]
            for i in psutil.net_if_addrs()
        }
        self.boot_time: float = psutil.boot_time()
        self.users: List[Dict[str, Any]] = [i._asdict() for i in psutil.users()]

        self.pid: int = os.getpid()
        self.proc: psutil.Process = psutil.Process(pid=self.pid)
        with self.proc.oneshot():  # perf
            self.proc_info: Dict[str, Any] = self.proc.as_dict(attrs=['exe', 'cmdline'])
            self.proc_child: List[psutil.Process] = self.proc.children(recursive=True)
            self.pid_child: List[int] = [p.pid for p in self.proc_child] + [self.pid]

        self.requirements: List[str] = []
        for dist in importlib.metadata.distributions():
            metadata = dist.metadata
            if not metadata:
                continue
            metadata_map = cast(Mapping[str, Any], metadata)
            name = metadata_map.get('Name')
            if isinstance(name, str):
                self.requirements.append(f'{name}=={dist.version}')
        if self.settings.mode == 'debug':  # privacy guard
            self.environ: Dict[str, str] = self.proc.environ()

        self.gpu: Dict[str, Any] = self.get_gpu()
        self.git: Dict[str, Any] = self.get_git()

    def __getattr__(self, name: str) -> Optional[Any]:
        return self.get_psutil(name)

    def get_psutil(self, name: str) -> Optional[Any]:  # handling os specific methods
        if hasattr(psutil, name):
            return getattr(psutil, name)
        else:
            return None

    def get_gpu(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}

        # NVIDIA
        n = run_cmd('nvidia-smi')
        if n:
            d['nvidia'] = {'smi': n}
            try:
                import pynvml

                try:
                    pynvml.nvmlInit()
                    logger.info(f'{tag}: NVIDIA GPU detected')
                    d['nvidia'].update(
                        {
                            'count': pynvml.nvmlDeviceGetCount(),
                            'driver': pynvml.nvmlSystemGetDriverVersion(),
                            'devices': [],
                            'handles': [],
                        }
                    )
                    for i in range(d['nvidia']['count']):
                        h = pynvml.nvmlDeviceGetHandleByIndex(i)
                        d['nvidia']['handles'].append(h)
                        d['nvidia']['devices'].append(
                            {
                                'name': pynvml.nvmlDeviceGetName(h),
                                'memory': {
                                    'total': pynvml.nvmlDeviceGetMemoryInfo(h).total
                                },
                                'temp': pynvml.nvmlDeviceGetTemperature(
                                    h, pynvml.NVML_TEMPERATURE_GPU
                                ),
                                'pid': [
                                    p.pid
                                    for p in (
                                        pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                                        + pynvml.nvmlDeviceGetGraphicsRunningProcesses(
                                            h
                                        )
                                    )
                                ],
                            }
                        )
                except pynvml.NVMLError_LibraryNotFound:
                    logger.debug(f'{tag}: NVIDIA: driver not found')
                except Exception as e:
                    logger.error('%s: NVIDIA: error: %s', tag, e)
            except ImportError:
                logger.debug(f'{tag}: NVIDIA: pynvml not found')
        return d

    def get_git(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        try:
            from git import Repo, exc
        except ImportError:
            logger.debug(f'{tag}: git: GitPython not installed, skipping git info')
            return d

        try:
            repo = Repo(
                f'{self.settings.dir}',
                search_parent_directories=True,
            )
            try:
                c = {
                    'name': repo.config_reader().get_value('user', 'name'),
                    'email': repo.config_reader().get_value('user', 'email'),
                }
            except Exception:
                c = {}
            try:
                c.update({'url': repo.remotes['origin'].url})
            except Exception:
                pass
            try:
                branch_name = repo.head.ref.name
            except TypeError:  # HEAD is detached
                branch_name = ''
            d = {
                'root': repo.git.rev_parse('--show-toplevel'),
                'dirty': repo.is_dirty(),
                'branch': branch_name or '',
                'commit': repo.head.commit.hexsha,
                **c,
            }
            if d['root']:
                cmd = 'git diff'
                if repo.git.version_info >= (2, 11, 0):  # TODO: remove legacy compat
                    cmd += ' --submodule=diff'
                d['diff'] = {
                    'remote': run_cmd(cmd + ' @{u}'),
                }
                if d['dirty']:
                    d['diff']['head'] = run_cmd(cmd + ' HEAD')
        except exc.InvalidGitRepositoryError:
            logger.debug(f'{tag}: git: not a git repository')
        except exc.GitError as e:
            logger.debug(
                '%s: git: repository not detected: (%s) %s',
                tag,
                e.__class__.__name__,
                e,
            )
        return d

    def _calc_rate(
        self,
        current: Dict[str, float],
        timestamp: float,
    ) -> Dict[str, float]:
        """
        Calculate rates (per-second) for counter metrics.

        Similar to Prometheus rate() function, computes the per-second rate
        of change between the current and previous sample.

        Args:
            current: Dict of current counter values
            timestamp: Current timestamp in seconds

        Returns:
            Dict of rate values (units per second)
        """
        rates: Dict[str, float] = {}

        if self._prev_timestamp is None:
            # First sample - store values but can't compute rate yet
            self._prev_counters = current.copy()
            self._prev_timestamp = timestamp
            return rates

        dt = timestamp - self._prev_timestamp
        if dt <= 0:
            return rates

        for key, value in current.items():
            prev_value = self._prev_counters.get(key)
            if prev_value is not None:
                # Handle counter reset (e.g., process restart or OS reset)
                if value >= prev_value:
                    delta = value - prev_value
                else:
                    # Counter reset to zero - use current value as delta (approximation)
                    delta = value
                rates[key] = delta / dt

        # Update previous values
        self._prev_counters = current.copy()
        self._prev_timestamp = timestamp

        return rates

    def get_infiniband_counters(self) -> Dict[str, int]:
        """
        Read InfiniBand port counters from sysfs.

        Returns counters for all InfiniBand devices and ports found at
        /sys/class/infiniband/*/ports/*/counters/

        Key counters:
        - port_rcv_data: Bytes received (raw is 4-byte words, converted)
        - port_xmit_data: Bytes transmitted (raw is 4-byte words, converted)
        - port_rcv_packets: Packets received
        - port_xmit_packets: Packets transmitted

        Returns:
            Dict with keys like 'ib.<device>.<port>.rcv_bytes', etc.
        """
        counters: Dict[str, int] = {}
        ib_base = '/sys/class/infiniband'

        if not os.path.exists(ib_base):
            return counters

        try:
            for device in os.listdir(ib_base):
                ports_dir = os.path.join(ib_base, device, 'ports')
                if not os.path.isdir(ports_dir):
                    continue

                for port in os.listdir(ports_dir):
                    counters_dir = os.path.join(ports_dir, port, 'counters')
                    if not os.path.isdir(counters_dir):
                        continue

                    prefix = f'ib.{device}.{port}'

                    # Read data and packet counters
                    counters_to_read = [
                        ('port_rcv_data', 'rcv_bytes', 4),
                        ('port_xmit_data', 'xmit_bytes', 4),
                        ('port_rcv_packets', 'rcv_packets', 1),
                        ('port_xmit_packets', 'xmit_packets', 1),
                    ]

                    for counter_name, metric_name, multiplier in counters_to_read:
                        counter_path = os.path.join(counters_dir, counter_name)
                        if os.path.exists(counter_path):
                            try:
                                with open(counter_path, 'r') as f:
                                    value = int(f.read().strip())
                                    key = f'{prefix}.{metric_name}'
                                    counters[key] = value * multiplier
                            except (IOError, ValueError) as e:
                                logger.debug(
                                    '%s: failed to read IB counter %s: %s',
                                    tag,
                                    counter_path,
                                    e,
                                )
        except OSError as e:
            logger.debug('%s: failed to enumerate InfiniBand devices: %s', tag, e)

        return counters

    def get_info(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            'process': {
                **self.proc_info,
                'pid': self.pid,
            },
            'platform': self.uname,
            'timezone': self.timezone,
            'cpu': {
                'physical': self.cpu_count(logical=False),
                'virtual': self.cpu_count(logical=True),
                'freq': {
                    'min': min([i['min'] for i in self.cpu_freq]),
                    'max': max([i['max'] for i in self.cpu_freq]),
                },
            },
            'memory': {
                'virt': self.svmem['total'],
                'swap': self.sswap['total'],
            },
            'boot_time': self.boot_time,
            'requirements': self.requirements,
        }
        if self.gpu:
            d['gpu'] = {}
            if self.gpu.get('nvidia'):
                d['gpu']['nvidia'] = {
                    k: v for k, v in self.gpu['nvidia'].items() if k != 'handles'
                }
        if self.git:
            d['git'] = self.git
        if self.settings.mode == 'debug':
            d['process']['environ'] = self.environ
            d = {
                **d,
                'disk': self.disk,
                'network': self.net_if_addrs,
                'users': self.users,
            }
        return d

    def monitor(self) -> Dict[str, Union[int, float]]:
        p = self.settings.x_sys_label
        now = time.time()

        # Collect network counters
        net_io = psutil.net_io_counters()._asdict()
        net_counters = {
            f'net.{k}': v for k, v in net_io.items() if k.startswith('bytes')
        }

        # Collect InfiniBand counters
        ib_counters = self.get_infiniband_counters()

        # Combine all counters for rate calculation
        all_counters: Dict[str, float] = {k: float(v) for k, v in net_counters.items()}
        all_counters.update({k: float(v) for k, v in ib_counters.items()})

        # Calculate rates (bytes/sec, packets/sec)
        rates = self._calc_rate(all_counters, now)

        # Build output dict
        d: Dict[str, Union[int, float]] = {
            **{
                f'{p}/cpu.pct.{i}': v
                for i, v in enumerate(psutil.cpu_percent(percpu=True))
            },
            **{
                f'{p}/mem.{k}': v
                for k, v in psutil.virtual_memory()._asdict().items()
                if k in ('active', 'used')
            },
            **{
                f'{p}/disk.{k}': v
                for k, v in psutil.disk_usage(self.settings.get_dir())._asdict().items()
                if k in ('used',)
            },
            # Raw network counters (cumulative)
            **{f'{p}/{k}': v for k, v in net_counters.items()},
            # InfiniBand counters (cumulative)
            **{f'{p}/{k}': v for k, v in ib_counters.items()},
            # Network and InfiniBand rates (bytes/sec or packets/sec)
            **{f'{p}/{k}.rate': v for k, v in rates.items()},
        }
        if self.gpu:
            if self.gpu.get('nvidia'):
                import pynvml

                for h in self.gpu['nvidia']['handles']:
                    idx = pynvml.nvmlDeviceGetIndex(h)
                    name = pynvml.nvmlDeviceGetName(h)
                    name = name.lower().replace(' ', '_').replace('-', '_')
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    d[f'{p}/gpu.{idx}.{name}.utilization'] = util.gpu
                    d[f'{p}/gpu.{idx}.{name}.memory.used'] = mem.used
                    d[f'{p}/gpu.{idx}.{name}.memory.utilization'] = util.memory
                    d[f'{p}/gpu.{idx}.{name}.power'] = pynvml.nvmlDeviceGetPowerUsage(h)
        return d

    @PendingDeprecationWarning
    def monitor_human(self) -> Dict[str, Any]:
        try:
            cpu_freq = [i.current for i in psutil.cpu_freq(percpu=True)]
        except Exception:  # errors on darwin t81xx
            cpu_freq = [0]

        d: Dict[str, Any] = {
            'cpu': {
                'percent': psutil.cpu_percent(percpu=True),
                'freq': cpu_freq,
            },
            'memory': {
                'virt': {
                    k: to_human(v)
                    for k, v in psutil.virtual_memory()._asdict().items()
                    if k != 'percent'
                },
            },
            'disk': {
                'out': to_human(psutil.disk_io_counters().read_bytes),
                'in': to_human(psutil.disk_io_counters().write_bytes),
                'usage': {
                    k: to_human(v)
                    for k, v in psutil.disk_usage(self.settings.get_dir())
                    ._asdict()
                    .items()
                    if k != 'percent'
                },
            },
            'network': {
                'out': to_human(psutil.net_io_counters().bytes_sent),
                'in': to_human(psutil.net_io_counters().bytes_recv),
            },
        }
        with self.proc.oneshot():  # perf
            d['process'] = {
                **self.proc.as_dict(
                    attrs=['status', 'cpu_percent', 'memory_percent', 'num_threads']
                ),
                'memory': to_human(self.proc.memory_info().rss),
            }
        if self.gpu:
            d['gpu'] = {}
            if self.gpu.get('nvidia'):
                import pynvml

                d['gpu']['nvidia'] = {}
                d['gpu']['nvidia']['devices'] = [
                    {
                        'name': pynvml.nvmlDeviceGetName(h),
                        'temp': pynvml.nvmlDeviceGetTemperature(
                            h, pynvml.NVML_TEMPERATURE_GPU
                        ),
                        'gpu_percent': pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
                        'memory': {
                            'percent': pynvml.nvmlDeviceGetUtilizationRates(h).memory,
                            'used': to_human(pynvml.nvmlDeviceGetMemoryInfo(h).used),
                            'total': to_human(pynvml.nvmlDeviceGetMemoryInfo(h).total),
                        },
                        'power': {
                            'usage': pynvml.nvmlDeviceGetPowerUsage(h),
                            'limit': pynvml.nvmlDeviceGetEnforcedPowerLimit(h),
                        },
                        'pid': [
                            p.pid
                            for p in (
                                pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                                + pynvml.nvmlDeviceGetGraphicsRunningProcesses(h)
                            )
                        ],
                    }
                    for h in self.gpu['nvidia']['handles']
                ]
        if self.settings.mode == 'debug':
            d['memory']['swap'] = {
                k: to_human(v)
                for k, v in psutil.swap_memory()._asdict().items()
                if k != 'percent'
            }
        return d
