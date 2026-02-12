import ctypes
import ctypes.util
import importlib.metadata
import logging
import os
import platform
import socket
import sys as _sys
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

    def get_konduktor(self) -> Dict[str, Any]:
        """Detect and collect Konduktor job metadata from environment."""
        job_name = os.environ.get('KONDUKTOR_JOB_NAME')
        if not job_name:
            return {}
        env_keys = {
            'num_nodes': 'NUM_NODES',
            'num_gpus_per_node': 'NUM_GPUS_PER_NODE',
            'rank': 'RANK',
            'master_addr': 'MASTER_ADDR',
            'accelerator_type': 'KONDUKTOR_ACCELERATOR_TYPE',
            'node_name': 'KONDUKTOR_NODENAME',
            'restart_attempt': 'RESTART_ATTEMPT',
            'namespace': 'KONDUKTOR_NAMESPACE',
        }
        d: Dict[str, Any] = {
            key: os.environ.get(env_var) for key, env_var in env_keys.items()
        }
        d['job_name'] = job_name
        # Cost-relevant: total GPU count for this job
        try:
            gpus = int(d.get('num_gpus_per_node') or 0)
            nodes = int(d.get('num_nodes') or 0)
            d['total_gpus'] = gpus * nodes
        except (ValueError, TypeError):
            pass
        return d

    def get_os_info(self) -> Dict[str, Any]:
        """Collect OS, kernel, Python, and glibc information."""
        d: Dict[str, Any] = {}

        # Hostname
        try:
            d['hostname'] = socket.gethostname()
        except Exception:
            d['hostname'] = platform.node()

        # OS / distribution info
        try:
            # platform.freedesktop_os_release() available in Python 3.10+
            if hasattr(platform, 'freedesktop_os_release'):
                os_release = platform.freedesktop_os_release()
                d['os'] = {
                    'name': os_release.get('NAME', ''),
                    'version': os_release.get(
                        'VERSION', os_release.get('BUILD_ID', '')
                    ),
                    'pretty_name': os_release.get('PRETTY_NAME', ''),
                    'id': os_release.get('ID', ''),
                }
            else:
                # Fallback: read /etc/os-release directly
                d['os'] = self._read_os_release()
        except Exception:
            d['os'] = {
                'name': platform.system(),
                'version': platform.version(),
            }

        # Linux kernel version
        d['kernel'] = platform.release()

        # Python version
        d['python'] = {
            'version': platform.python_version(),
            'implementation': platform.python_implementation(),
            'compiler': platform.python_compiler(),
            'executable': _sys.executable,
        }

        # glibc version
        try:
            glibc = platform.libc_ver()
            if glibc[0]:
                d['glibc'] = glibc[1]
            else:
                # Fallback: use ctypes to query glibc directly
                libc_name = ctypes.util.find_library('c')
                if libc_name:
                    libc = ctypes.CDLL(libc_name)
                    gnu_get_libc_version = libc.gnu_get_libc_version
                    gnu_get_libc_version.restype = ctypes.c_char_p
                    d['glibc'] = gnu_get_libc_version().decode('ascii')
        except Exception:
            pass

        return d

    def _read_os_release(self) -> Dict[str, str]:
        """Fallback to read /etc/os-release for Python < 3.10."""
        d: Dict[str, str] = {}
        for path in ('/etc/os-release', '/usr/lib/os-release'):
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if '=' in line:
                                key, _, value = line.partition('=')
                                value = value.strip('"').strip("'")
                                if key in ('NAME', 'VERSION', 'PRETTY_NAME', 'ID'):
                                    d[key.lower()] = value
                except Exception:
                    pass
                break
        return d

    def get_cuda_info(self) -> Dict[str, Any]:
        """Collect CUDA and cuDNN version information from system and PyTorch."""
        d: Dict[str, Any] = {}

        # System CUDA version from nvidia-smi
        try:
            nvcc_out = run_cmd('nvcc --version')
            if nvcc_out:
                import re

                m = re.search(r'release\s+([\d.]+)', nvcc_out)
                if m:
                    d['cuda_nvcc'] = m.group(1)
        except Exception:
            pass

        # CUDA version reported by nvidia-smi (driver-supported max)
        if self.gpu.get('nvidia', {}).get('smi'):
            try:
                import re

                m = re.search(r'CUDA Version:\s+([\d.]+)', self.gpu['nvidia']['smi'])
                if m:
                    d['cuda_driver'] = m.group(1)
            except Exception:
                pass

        # PyTorch CUDA/cuDNN versions
        try:
            import torch

            d['cuda_pytorch'] = getattr(torch.version, 'cuda', None)
            try:
                d['cudnn_pytorch'] = str(torch.backends.cudnn.version())
            except Exception:
                pass
        except ImportError:
            pass

        # CUDA environment variables
        cuda_env: Dict[str, str] = {}
        for key, value in os.environ.items():
            if key.startswith('CUDA_'):
                cuda_env[key] = value
        if cuda_env:
            d['cuda_env'] = cuda_env

        return d

    def get_nccl_info(self) -> Dict[str, Any]:
        """Collect NCCL version information and NCCL environment variables."""
        d: Dict[str, Any] = {}

        # PyTorch NCCL version
        try:
            import torch

            if hasattr(torch.cuda, 'nccl') and hasattr(torch.cuda.nccl, 'version'):
                nccl_ver = torch.cuda.nccl.version()
                if isinstance(nccl_ver, tuple):
                    d['nccl_pytorch'] = '.'.join(str(x) for x in nccl_ver)
                else:
                    d['nccl_pytorch'] = str(nccl_ver)
        except Exception:
            pass

        # System NCCL version: check the shared library
        try:
            libnccl = ctypes.util.find_library('nccl')
            if libnccl:
                nccl_lib = ctypes.CDLL(libnccl)
                # ncclGetVersion returns an int: major*10000 + minor*100 + patch
                nccl_get_version = nccl_lib.ncclGetVersion
                version = ctypes.c_int()
                result = nccl_get_version(ctypes.byref(version))
                if result == 0:  # ncclSuccess
                    v = version.value
                    major = v // 10000
                    minor = (v % 10000) // 100
                    patch = v % 100
                    d['nccl_system'] = f'{major}.{minor}.{patch}'
        except Exception:
            pass

        # NCCL environment variables
        nccl_env: Dict[str, str] = {}
        for key, value in os.environ.items():
            if key.startswith('NCCL_'):
                nccl_env[key] = value
        if nccl_env:
            d['nccl_env'] = nccl_env

        return d

    def get_infiniband_info(self) -> Dict[str, Any]:
        """Collect InfiniBand device info: link type, speed, mode (IB vs RoCE)."""
        d: Dict[str, Any] = {}
        ib_base = '/sys/class/infiniband'

        if not os.path.exists(ib_base):
            return d

        try:
            devices: List[Dict[str, Any]] = []
            for device in sorted(os.listdir(ib_base)):
                dev_info: Dict[str, Any] = {'name': device}

                # Read board_id / fw_ver if available
                for attr in ('board_id', 'fw_ver', 'hca_type'):
                    attr_path = os.path.join(ib_base, device, attr)
                    if os.path.exists(attr_path):
                        try:
                            with open(attr_path, 'r') as f:
                                dev_info[attr] = f.read().strip()
                        except Exception:
                            pass

                # Enumerate ports
                ports_dir = os.path.join(ib_base, device, 'ports')
                if os.path.isdir(ports_dir):
                    ports: List[Dict[str, str]] = []
                    for port in sorted(os.listdir(ports_dir)):
                        port_info: Dict[str, str] = {'port': port}
                        port_dir = os.path.join(ports_dir, port)

                        # link_layer: InfiniBand or Ethernet (RoCE)
                        for attr in ('link_layer', 'state', 'rate', 'phys_state'):
                            attr_path = os.path.join(port_dir, attr)
                            if os.path.exists(attr_path):
                                try:
                                    with open(attr_path, 'r') as f:
                                        port_info[attr] = f.read().strip()
                                except Exception:
                                    pass

                        # GID table entry 0 for RoCE mode detection
                        gid_path = os.path.join(port_dir, 'gids', '0')
                        if os.path.exists(gid_path):
                            try:
                                with open(gid_path, 'r') as f:
                                    gid = f.read().strip()
                                    empty = '0000:0000:0000:0000:0000:0000:0000:0000'
                                    if gid and gid != empty:
                                        port_info['gid_0'] = gid
                            except Exception:
                                pass

                        # RoCE type from gid_attrs if available
                        gid_type_path = os.path.join(
                            port_dir, 'gid_attrs', 'types', '0'
                        )
                        if os.path.exists(gid_type_path):
                            try:
                                with open(gid_type_path, 'r') as f:
                                    port_info['roce_type'] = f.read().strip()
                            except Exception:
                                pass

                        ports.append(port_info)
                    dev_info['ports'] = ports

                devices.append(dev_info)

            if devices:
                d['devices'] = devices
        except OSError as e:
            logger.debug('%s: failed to enumerate InfiniBand info: %s', tag, e)

        return d

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

        # OS, kernel, Python, glibc, hostname
        try:
            os_info = self.get_os_info()
            if os_info:
                d['os_info'] = os_info
        except Exception as e:
            logger.debug('%s: failed to collect OS info: %s', tag, e)

        # CUDA / cuDNN versions
        try:
            cuda_info = self.get_cuda_info()
            if cuda_info:
                d['cuda'] = cuda_info
        except Exception as e:
            logger.debug('%s: failed to collect CUDA info: %s', tag, e)

        # NCCL versions and environment variables
        try:
            nccl_info = self.get_nccl_info()
            if nccl_info:
                d['nccl'] = nccl_info
        except Exception as e:
            logger.debug('%s: failed to collect NCCL info: %s', tag, e)

        # InfiniBand / RoCE device info
        try:
            ib_info = self.get_infiniband_info()
            if ib_info:
                d['infiniband'] = ib_info
        except Exception as e:
            logger.debug('%s: failed to collect InfiniBand info: %s', tag, e)

        if self.gpu:
            d['gpu'] = {}
            if self.gpu.get('nvidia'):
                d['gpu']['nvidia'] = {
                    k: v for k, v in self.gpu['nvidia'].items() if k != 'handles'
                }
        if self.git:
            d['git'] = self.git
        konduktor = self.get_konduktor()
        if konduktor:
            d['konduktor'] = konduktor
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
