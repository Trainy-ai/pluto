"""
Example: Migrating from Neptune to mlop with dual-logging monkeypatch.

This example demonstrates how to use the Neptune-to-mlop compatibility layer
for a seamless migration during the transition period.

## Quick Start

### Step 1: Add one import line
Simply add this import at the top of your existing Neptune training script:

```python
import mlop.compat.neptune  # Enable dual-logging to mlop
```

That's it! Your existing Neptune code will now log to both Neptune and mlop.

### Step 2: Configure mlop (via environment variables)
Set these environment variables before running your script:

```bash
export MLOP_PROJECT="my-project-name"
export MLOP_API_KEY="your-api-key"  # Optional, falls back to keyring
```

### Step 3: Run your existing Neptune script
No code changes needed! Your script will dual-log to both systems.

### Step 4 (After transition): Remove Neptune
Once you've migrated fully to mlop, you can:
1. Remove the `import mlop.compat.neptune` line
2. Replace Neptune imports with mlop
3. Update API calls to use mlop directly

## Example Training Script (Before Migration)
"""


# Original Neptune training script (unchanged during transition)
def original_neptune_training_script():
    """
    This is what your original Neptune training script might look like.
    During the transition period, you only need to add the compatibility import.
    """
    from neptune_scale import Run

    # Initialize Neptune run
    run = Run(
        experiment_name='my-training-experiment',
        project='my-team/my-project',
    )

    # Log hyperparameters
    run.log_configs(
        {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'model': 'resnet50',
            'optimizer': 'adam',
        }
    )

    # Add tags
    run.add_tags(['baseline', 'experiment-v1'])

    # Simulate training loop
    for epoch in range(100):
        # Training step (simplified)
        train_loss = 1.0 / (epoch + 1)
        train_acc = min(0.99, epoch / 100)

        # Log metrics
        run.log_metrics(
            {
                'train/loss': train_loss,
                'train/accuracy': train_acc,
            },
            step=epoch,
        )

        # Validation step
        if epoch % 10 == 0:
            val_loss = train_loss * 1.1
            val_acc = train_acc * 0.95

            run.log_metrics(
                {
                    'val/loss': val_loss,
                    'val/accuracy': val_acc,
                },
                step=epoch,
            )

    # Close the run
    run.close()

    print('Training completed!')


# Migrated script (with dual-logging enabled)
def migrated_dual_logging_script():
    """
    This is the same script with dual-logging enabled.
    Only difference: added one import line at the top.
    """
    # ADD THIS LINE to enable dual-logging to mlop
    # CRITICAL: This import MUST come before Neptune imports for monkeypatch to work
    import mlop.compat.neptune  # noqa: F401, I001

    from neptune_scale import Run

    # Rest of the code is IDENTICAL to the original
    run = Run(
        experiment_name='my-training-experiment',
        project='my-team/my-project',
    )

    run.log_configs(
        {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'model': 'resnet50',
            'optimizer': 'adam',
        }
    )

    run.add_tags(['baseline', 'experiment-v1'])

    for epoch in range(100):
        train_loss = 1.0 / (epoch + 1)
        train_acc = min(0.99, epoch / 100)

        run.log_metrics(
            {
                'train/loss': train_loss,
                'train/accuracy': train_acc,
            },
            step=epoch,
        )

        if epoch % 10 == 0:
            val_loss = train_loss * 1.1
            val_acc = train_acc * 0.95

            run.log_metrics(
                {
                    'val/loss': val_loss,
                    'val/accuracy': val_acc,
                },
                step=epoch,
            )

    run.close()

    print('Training completed! Logged to both Neptune and mlop.')


# After full migration to mlop
def fully_migrated_mlop_script():
    """
    This is what your script might look like after full migration to mlop.
    API is similar but uses mlop directly.
    """
    import mlop

    # Initialize mlop run
    run = mlop.init(
        project='my-project-name',
        name='my-training-experiment',
        config={
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'model': 'resnet50',
            'optimizer': 'adam',
        },
    )

    # Note: mlop auto-increments steps, so no need to pass step explicitly
    for epoch in range(100):
        train_loss = 1.0 / (epoch + 1)
        train_acc = min(0.99, epoch / 100)

        # Log metrics (step auto-increments)
        run.log(
            {
                'train/loss': train_loss,
                'train/accuracy': train_acc,
            }
        )

        if epoch % 10 == 0:
            val_loss = train_loss * 1.1
            val_acc = train_acc * 0.95

            run.log(
                {
                    'val/loss': val_loss,
                    'val/accuracy': val_acc,
                }
            )

    run.finish()

    print('Training completed with mlop!')


# Example with image logging
def image_logging_example():
    """
    Example showing how image logging works during migration.
    """
    # CRITICAL: mlop.compat.neptune MUST be imported before Neptune imports
    import mlop.compat.neptune  # noqa: F401, I001

    import tempfile
    from pathlib import Path

    import numpy as np
    from neptune_scale import Run
    from neptune_scale.types import File

    try:
        from PIL import Image
    except ImportError:
        print('PIL not available, skipping image logging example')
        return

    run = Run(experiment_name='image-logging-test')

    # Create a temporary directory for dummy images
    with tempfile.TemporaryDirectory() as tmpdir:
        # Log an image using Neptune's File API
        # This will be automatically converted to mlop.Image
        # Create a dummy image (64x64 RGB)
        dummy_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img_path = Path(tmpdir) / 'sample_image.png'
        Image.fromarray(dummy_img).save(img_path)

        sample_image = File(source=str(img_path), mime_type='image/png')
        run.assign_files({'samples/image1': sample_image})

        # Or log a series of images
        for i in range(5):
            # Create unique dummy images for each step
            dummy_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            image_path = Path(tmpdir) / f'image_{i}.png'
            Image.fromarray(dummy_img).save(image_path)

            run.log_files({'training/samples': File(source=str(image_path))}, step=i)

    run.close()


# Example with histogram logging
def histogram_logging_example():
    """
    Example showing how histogram logging works during migration.
    """
    # CRITICAL: mlop.compat.neptune MUST be imported before Neptune imports
    import mlop.compat.neptune  # noqa: F401, I001

    import numpy as np
    from neptune_scale import Run
    from neptune_scale.types import Histogram

    run = Run(experiment_name='histogram-test')

    # Log histograms (e.g., layer activations)
    for layer_idx in range(3):
        # Simulate layer activations
        activations = np.random.randn(1000)

        # Create histogram
        counts, bin_edges = np.histogram(activations, bins=50)
        hist = Histogram(bin_edges=bin_edges, counts=counts)

        run.log_histograms({f'layer{layer_idx}/activations': hist}, step=layer_idx)

    run.close()


# Example configuration in different environments
def configuration_examples():
    """
    Examples of how to configure mlop in different environments.
    """

    # Example 1: Development environment (using keyring)
    # No environment variables needed - mlop will use stored credentials
    # Just set MLOP_PROJECT:
    #   export MLOP_PROJECT="dev-project"

    # Example 2: CI/CD environment (using env vars)
    # Set all credentials via environment variables:
    #   export MLOP_PROJECT="ci-project"
    #   export MLOP_API_KEY="your-ci-api-key"

    # Example 3: Docker container
    # Pass environment variables when running the container:
    #   docker run -e MLOP_PROJECT="docker-project" \
    #              -e MLOP_API_KEY="your-key" \
    #              your-image

    # Example 4: Custom mlop instance
    # Specify custom URLs for self-hosted mlop:
    #   export MLOP_PROJECT="custom-project"
    #   export MLOP_URL_APP="https://mlop.yourcompany.com"
    #   export MLOP_URL_API="https://mlop-api.yourcompany.com"
    #   export MLOP_URL_INGEST="https://mlop-ingest.yourcompany.com"

    pass


if __name__ == '__main__':
    """
    Run examples based on command-line argument.
    """
    import sys

    if len(sys.argv) < 2:
        print('Usage: python neptune_migration_example.py <example>')
        print('Examples:')
        print('  original    - Run original Neptune script (no dual-logging)')
        print('  migrated    - Run with dual-logging enabled')
        print('  mlop        - Run fully migrated to mlop')
        print('  image       - Image logging example')
        print('  histogram   - Histogram logging example')
        sys.exit(1)

    example = sys.argv[1]

    if example == 'original':
        original_neptune_training_script()
    elif example == 'migrated':
        migrated_dual_logging_script()
    elif example == 'mlop':
        fully_migrated_mlop_script()
    elif example == 'image':
        image_logging_example()
    elif example == 'histogram':
        histogram_logging_example()
    else:
        print(f'Unknown example: {example}')
        sys.exit(1)
