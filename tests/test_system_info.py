"""Tests for system info collection: OS, CUDA, NCCL, InfiniBand."""

import os
import platform
import sys
from unittest.mock import patch

import pytest

from pluto.sys import System


class TestSystemInfoHelper:
    """Shared helper for creating System instances."""

    def _make_system(self):
        from pluto.sets import setup

        settings = setup(None)
        return System(settings)


class TestGetOsInfo(TestSystemInfoHelper):
    """Tests for System.get_os_info() method."""

    def test_returns_hostname(self):
        sys_obj = self._make_system()
        result = sys_obj.get_os_info()
        assert 'hostname' in result
        assert isinstance(result['hostname'], str)
        assert len(result['hostname']) > 0

    def test_returns_kernel(self):
        sys_obj = self._make_system()
        result = sys_obj.get_os_info()
        assert 'kernel' in result
        assert result['kernel'] == platform.release()

    def test_returns_python_info(self):
        sys_obj = self._make_system()
        result = sys_obj.get_os_info()
        assert 'python' in result
        py = result['python']
        assert py['version'] == platform.python_version()
        assert py['implementation'] == platform.python_implementation()
        assert py['compiler'] == platform.python_compiler()
        assert py['executable'] == sys.executable

    def test_returns_os_info(self):
        sys_obj = self._make_system()
        result = sys_obj.get_os_info()
        assert 'os' in result
        # Should have at least a name field
        os_info = result['os']
        assert isinstance(os_info, dict)
        assert len(os_info) > 0

    def test_returns_glibc_on_linux(self):
        if sys.platform != 'linux':
            pytest.skip('glibc only on Linux')
        sys_obj = self._make_system()
        result = sys_obj.get_os_info()
        assert 'glibc' in result
        assert isinstance(result['glibc'], str)
        # Should look like a version string (e.g. "2.39")
        parts = result['glibc'].split('.')
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts)

    def test_get_info_includes_os_info(self):
        sys_obj = self._make_system()
        info = sys_obj.get_info()
        assert 'os_info' in info
        assert 'hostname' in info['os_info']
        assert 'python' in info['os_info']
        assert 'kernel' in info['os_info']

    def test_read_os_release_fallback(self):
        """_read_os_release reads /etc/os-release when available."""
        sys_obj = self._make_system()
        if os.path.exists('/etc/os-release'):
            result = sys_obj._read_os_release()
            assert isinstance(result, dict)
            # On a real Linux, should have at least 'name'
            if result:
                assert any(k in result for k in ('name', 'pretty_name', 'id'))
        else:
            result = sys_obj._read_os_release()
            assert result == {}


class TestGetCudaInfo(TestSystemInfoHelper):
    """Tests for System.get_cuda_info() method."""

    def test_returns_dict(self):
        sys_obj = self._make_system()
        result = sys_obj.get_cuda_info()
        assert isinstance(result, dict)

    def test_cuda_pytorch_when_torch_available(self):
        """If torch is importable, cuda_pytorch should be present."""
        try:
            import torch

            if torch.version.cuda is not None:
                sys_obj = self._make_system()
                result = sys_obj.get_cuda_info()
                assert 'cuda_pytorch' in result
                assert isinstance(result['cuda_pytorch'], str)
            else:
                pytest.skip('torch built without CUDA')
        except ImportError:
            pytest.skip('torch not installed')

    def test_cudnn_pytorch_when_available(self):
        """If torch has cuDNN, cudnn_pytorch should be a string."""
        try:
            import torch

            if torch.backends.cudnn.is_available():
                sys_obj = self._make_system()
                result = sys_obj.get_cuda_info()
                assert 'cudnn_pytorch' in result
                assert isinstance(result['cudnn_pytorch'], str)
            else:
                pytest.skip('cuDNN not available')
        except ImportError:
            pytest.skip('torch not installed')

    def test_collects_cuda_env_vars(self):
        """CUDA_* env vars are captured."""
        env = {'CUDA_VISIBLE_DEVICES': '0,1', 'CUDA_LAUNCH_BLOCKING': '1'}
        with patch.dict(os.environ, env, clear=False):
            sys_obj = self._make_system()
            result = sys_obj.get_cuda_info()
            assert 'cuda_env' in result
            assert result['cuda_env']['CUDA_VISIBLE_DEVICES'] == '0,1'
            assert result['cuda_env']['CUDA_LAUNCH_BLOCKING'] == '1'

    def test_no_cuda_env_when_none_set(self):
        """cuda_env key absent when no CUDA_* vars exist."""
        env = {k: v for k, v in os.environ.items() if not k.startswith('CUDA_')}
        with patch.dict(os.environ, env, clear=True):
            sys_obj = self._make_system()
            result = sys_obj.get_cuda_info()
            assert 'cuda_env' not in result

    def test_get_info_includes_cuda_when_available(self):
        sys_obj = self._make_system()
        info = sys_obj.get_info()
        result = sys_obj.get_cuda_info()
        if result:
            assert 'cuda' in info
        # If no CUDA at all, key may be absent - that's fine


class TestGetNcclInfo(TestSystemInfoHelper):
    """Tests for System.get_nccl_info() method."""

    def test_returns_dict(self):
        sys_obj = self._make_system()
        result = sys_obj.get_nccl_info()
        assert isinstance(result, dict)

    def test_nccl_pytorch_when_available(self):
        """If torch.cuda.nccl is available, nccl_pytorch should be present."""
        try:
            import torch

            if hasattr(torch.cuda, 'nccl') and hasattr(torch.cuda.nccl, 'version'):
                sys_obj = self._make_system()
                result = sys_obj.get_nccl_info()
                assert 'nccl_pytorch' in result
                # Should be a dotted version string like "2.26.2"
                parts = result['nccl_pytorch'].split('.')
                assert len(parts) >= 2
            else:
                pytest.skip('torch.cuda.nccl not available')
        except ImportError:
            pytest.skip('torch not installed')

    def test_collects_nccl_env_vars(self):
        """NCCL_* env vars are captured."""
        env = {
            'NCCL_DEBUG': 'INFO',
            'NCCL_SOCKET_IFNAME': 'eth0',
            'NCCL_IB_DISABLE': '0',
        }
        with patch.dict(os.environ, env, clear=False):
            sys_obj = self._make_system()
            result = sys_obj.get_nccl_info()
            assert 'nccl_env' in result
            assert result['nccl_env']['NCCL_DEBUG'] == 'INFO'
            assert result['nccl_env']['NCCL_SOCKET_IFNAME'] == 'eth0'
            assert result['nccl_env']['NCCL_IB_DISABLE'] == '0'

    def test_no_nccl_env_when_none_set(self):
        """nccl_env key absent when no NCCL_* vars exist."""
        env = {k: v for k, v in os.environ.items() if not k.startswith('NCCL_')}
        with patch.dict(os.environ, env, clear=True):
            sys_obj = self._make_system()
            result = sys_obj.get_nccl_info()
            assert 'nccl_env' not in result

    def test_get_info_includes_nccl_when_available(self):
        sys_obj = self._make_system()
        info = sys_obj.get_info()
        result = sys_obj.get_nccl_info()
        if result:
            assert 'nccl' in info


class TestGetInfinibandInfo(TestSystemInfoHelper):
    """Tests for System.get_infiniband_info() method."""

    def test_returns_dict(self):
        sys_obj = self._make_system()
        result = sys_obj.get_infiniband_info()
        assert isinstance(result, dict)

    def test_empty_when_no_ib_sysfs(self):
        """Returns empty dict when /sys/class/infiniband doesn't exist."""
        with patch('os.path.exists', return_value=False):
            sys_obj = self._make_system()
            result = sys_obj.get_infiniband_info()
            assert result == {}

    def test_parses_ib_device_structure(self):
        """Validates structure when IB devices are present."""
        sys_obj = self._make_system()
        result = sys_obj.get_infiniband_info()
        if not result:
            pytest.skip('No InfiniBand devices on this machine')
        assert 'devices' in result
        for dev in result['devices']:
            assert 'name' in dev
            if 'ports' in dev:
                for port in dev['ports']:
                    assert 'port' in port

    def test_get_info_includes_infiniband_when_available(self):
        sys_obj = self._make_system()
        info = sys_obj.get_info()
        ib = sys_obj.get_infiniband_info()
        if ib:
            assert 'infiniband' in info
        else:
            assert 'infiniband' not in info

    def test_mock_ib_sysfs(self, tmp_path):
        """Test IB parsing with a mocked sysfs tree."""
        ib_base = tmp_path
        dev_dir = ib_base / 'mlx5_0'
        port_dir = dev_dir / 'ports' / '1'
        gid_dir = port_dir / 'gids'
        gid_attrs_dir = port_dir / 'gid_attrs' / 'types'
        gid_dir.mkdir(parents=True)
        gid_attrs_dir.mkdir(parents=True)

        # Device attributes
        (dev_dir / 'fw_ver').write_text('20.31.1014\n')
        (dev_dir / 'board_id').write_text('MT_0000000223\n')
        (dev_dir / 'hca_type').write_text('ConnectX-6\n')

        # Port attributes
        (port_dir / 'link_layer').write_text('InfiniBand\n')
        (port_dir / 'state').write_text('4: ACTIVE\n')
        (port_dir / 'rate').write_text('200 Gb/sec (4X HDR)\n')
        (port_dir / 'phys_state').write_text('5: LinkUp\n')

        # GID info for RoCE detection
        (gid_dir / '0').write_text('fe80:0000:0000:0000:ec0d:9a03:0078:a8b2\n')
        (gid_attrs_dir / '0').write_text('RoCE v2\n')

        sys_obj = self._make_system()
        result = sys_obj.get_infiniband_info(ib_base=str(ib_base))

        assert 'devices' in result
        assert len(result['devices']) == 1
        dev = result['devices'][0]
        assert dev['name'] == 'mlx5_0'
        assert dev['fw_ver'] == '20.31.1014'
        assert dev['board_id'] == 'MT_0000000223'
        assert dev['hca_type'] == 'ConnectX-6'
        assert 'ports' in dev
        assert len(dev['ports']) == 1
        port = dev['ports'][0]
        assert port['port'] == '1'
        assert port['link_layer'] == 'InfiniBand'
        assert port['gid_0'] == 'fe80:0000:0000:0000:ec0d:9a03:0078:a8b2'
        assert port['roce_type'] == 'RoCE v2'


class TestGetInfoIntegration(TestSystemInfoHelper):
    """Integration tests for get_info() including all new fields."""

    def test_get_info_returns_os_info(self):
        sys_obj = self._make_system()
        info = sys_obj.get_info()
        assert 'os_info' in info
        os_info = info['os_info']
        assert 'hostname' in os_info
        assert 'kernel' in os_info
        assert 'python' in os_info
        assert 'executable' in os_info['python']

    def test_get_info_all_new_keys_are_dicts(self):
        """All new top-level keys should be dicts when present."""
        sys_obj = self._make_system()
        info = sys_obj.get_info()
        for key in ('os_info', 'cuda', 'nccl', 'infiniband'):
            if key in info:
                assert isinstance(info[key], dict), f'{key} should be a dict'

    def test_get_info_still_has_legacy_keys(self):
        """Ensure new fields didn't break existing ones."""
        sys_obj = self._make_system()
        info = sys_obj.get_info()
        for key in (
            'process',
            'platform',
            'timezone',
            'cpu',
            'memory',
            'boot_time',
            'requirements',
        ):
            assert key in info, f'Missing legacy key: {key}'

    def test_get_info_serializable(self):
        """get_info() result must be JSON-serializable."""
        import json

        sys_obj = self._make_system()
        info = sys_obj.get_info()
        # Remove handles (not serializable) if present
        if 'gpu' in info and 'nvidia' in info.get('gpu', {}):
            info['gpu']['nvidia'].pop('handles', None)
        serialized = json.dumps(info)
        assert isinstance(serialized, str)
        assert len(serialized) > 0

    def test_collection_failures_dont_crash(self):
        """If a collection method raises, get_info() still succeeds."""
        sys_obj = self._make_system()
        with (
            patch.object(sys_obj, 'get_os_info', side_effect=RuntimeError('boom')),
            patch.object(sys_obj, 'get_cuda_info', side_effect=RuntimeError('boom')),
            patch.object(sys_obj, 'get_nccl_info', side_effect=RuntimeError('boom')),
            patch.object(
                sys_obj, 'get_infiniband_info', side_effect=RuntimeError('boom')
            ),
        ):
            info = sys_obj.get_info()
            # Should still have the base keys
            assert 'process' in info
            assert 'platform' in info
            # New keys should be absent (collection failed)
            assert 'os_info' not in info
            assert 'cuda' not in info
            assert 'nccl' not in info
            assert 'infiniband' not in info
