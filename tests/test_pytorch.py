import os
import socket
from contextlib import closing

import pytest

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except Exception:  # pragma: no cover - optional dependency
        FSDP = None  # type: ignore[assignment, misc]

    HAS_TORCH = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    nn = optim = DataLoader = Subset = datasets = transforms = None  # type: ignore[assignment]
    DDP = FSDP = None  # type: ignore[assignment]

import pluto
from tests.utils import get_task_name

TESTING_PROJECT_NAME = 'testing-ci'
NUM_EPOCHS = 1
BATCH_SIZE = 64
NUM_WORKERS = 0
LEARNING_RATE = 0.005
MAX_BATCHES_PER_EPOCH = 5


if HAS_TORCH:

    class SimpleCNN(nn.Module):  # pragma: no cover - exercised via integration test
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(32 * 14 * 14, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            return x

    def _evaluate_model(model, data_loader, criterion):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, labels in data_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        avg_loss = total_loss / max(len(data_loader), 1)
        model.train()
        return avg_loss
else:  # pragma: no cover - executed only when torch missing

    class SimpleCNN:  # type: ignore[no-redef]
        pass

    def _evaluate_model(model, data_loader, criterion):  # type: ignore[no-redef]
        raise RuntimeError('torch not available')


def _world_size() -> int:
    return int(os.environ.get('WORLD_SIZE', '1'))


def _maybe_init_process_group() -> None:
    if dist is None:
        return
    if dist.is_initialized():
        return
    if 'MASTER_ADDR' in os.environ:
        dist.init_process_group(backend='gloo', init_method='env://')
    else:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        dist.init_process_group(
            backend='gloo',
            init_method=f'tcp://127.0.0.1:{port}',
            world_size=1,
            rank=0,
        )


@pytest.mark.skipif(not HAS_TORCH, reason='torch/torchvision not installed')
def test_pytorch_cnn_mnist_trains_and_logs():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    try:
        dataset = datasets.MNIST(
            root='./tests',
            train=True,
            transform=transform,
            download=True,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f'Unable to prepare MNIST dataset: {exc}')

    subset_size = min(len(dataset), BATCH_SIZE * MAX_BATCHES_PER_EPOCH)
    subset = Subset(dataset, list(range(subset_size)))
    train_loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    run = pluto.init(
        project=TESTING_PROJECT_NAME,
        name=get_task_name(),
        config={
            'epochs': NUM_EPOCHS,
            'lr': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'max_batches': MAX_BATCHES_PER_EPOCH,
        },
    )

    pluto.watch(model, disable_graph=False, freq=100)

    try:
        for epoch in range(NUM_EPOCHS):
            total_loss = 0.0
            for batch_idx, (images, labels) in enumerate(train_loader):
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                run.log({'loss/batch': loss.item(), 'batch_idx': batch_idx})
                if batch_idx + 1 >= MAX_BATCHES_PER_EPOCH:
                    break

            avg_loss = total_loss / max(
                1, min(len(train_loader), MAX_BATCHES_PER_EPOCH)
            )
            val_loss = _evaluate_model(model, train_loader, criterion)
            run.log(
                {
                    'epoch': epoch + 1,
                    'loss/train': avg_loss,
                    'loss/val': val_loss,
                }
            )
    finally:
        run.finish()


@pytest.mark.skipif(not HAS_TORCH, reason='torch/torchvision not installed')
def test_pluto_watch_tracks_convnet_gradients():
    class ConvNet(nn.Module):
        def __init__(self, kernels, classes=10):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(kernels[0], kernels[1], kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.fc = nn.Linear(7 * 7 * kernels[-1], classes)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    model = ConvNet([16, 32], 10)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    run = pluto.init(
        project=TESTING_PROJECT_NAME,
        name=get_task_name(),
        config={'test': 'watch-api'},
    )

    pluto.watch(model, disable_graph=False, freq=100)

    try:
        for step in range(3):
            images = torch.randn(8, 1, 28, 28)
            labels = torch.randint(0, 10, (8,))
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run.log({'watch/loss': loss.item(), 'watch/step': step})
    finally:
        run.finish()


@pytest.mark.distributed
@pytest.mark.skipif(not HAS_TORCH, reason='torch/torchvision not installed')
@pytest.mark.skipif(
    DDP is None or dist is None or not dist.is_available(),
    reason='torch.distributed not available',
)
def test_pluto_watch_on_ddp_model():
    if _world_size() < 2:
        pytest.skip('DDP watch test requires WORLD_SIZE >= 2 (run via torchrun)')

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    if dist.is_initialized():
        dist.destroy_process_group()
    _maybe_init_process_group()
    rank = dist.get_rank() if dist.is_initialized() else 0

    model = SimpleCNN()
    ddp_model = DDP(model)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    run = pluto.init(
        project=TESTING_PROJECT_NAME,
        name=f'{get_task_name()}-rank-{rank}',
        config={'test': 'watch-ddp', 'rank': rank, 'world_size': _world_size()},
    )
    pluto.watch(ddp_model, disable_graph=False, freq=50)

    try:
        for step in range(2):
            images = torch.randn(4, 1, 28, 28)
            labels = torch.randint(0, 10, (4,))
            outputs = ddp_model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run.log(
                {f'rank.{rank}/ddp/loss': loss.item(), f'rank.{rank}/ddp/step': step}
            )
        if dist.is_initialized():
            dist.barrier()
    finally:
        run.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.skipif(not HAS_TORCH, reason='torch/torchvision not installed')
@pytest.mark.skipif(
    FSDP is None or dist is None or not dist.is_available(),
    reason='torch.distributed.fsdp not available',
)
def test_pluto_watch_on_fsdp_model():
    if _world_size() < 2:
        pytest.skip('FSDP watch test requires WORLD_SIZE >= 2 (run via torchrun)')

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    if dist.is_initialized():
        dist.destroy_process_group()
    _maybe_init_process_group()
    rank = dist.get_rank() if dist.is_initialized() else 0

    model = SimpleCNN()
    fsdp_model = FSDP(model)  # type: ignore[arg-type]
    optimizer = optim.SGD(fsdp_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    run = pluto.init(
        project=TESTING_PROJECT_NAME,
        name=f'{get_task_name()}-rank-{rank}',
        config={'test': 'watch-fsdp', 'rank': rank, 'world_size': _world_size()},
    )
    pluto.watch(fsdp_model, disable_graph=False, freq=50)

    try:
        for step in range(2):
            images = torch.randn(4, 1, 28, 28)
            labels = torch.randint(0, 10, (4,))
            outputs = fsdp_model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run.log({'fsdp/loss': loss.item(), 'fsdp/step': step, 'rank': rank})
        if dist.is_initialized():
            dist.barrier()
    finally:
        run.finish()
        if dist.is_initialized():
            dist.destroy_process_group()
