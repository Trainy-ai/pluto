[![pypi](https://img.shields.io/pypi/v/pluto-ml)](https://pypi.org/project/pluto-ml/)

**Pluto** is an experiment tracking platform. It provides [self-hostable superior experimental tracking capabilities and lifecycle management for training ML models](https://docs.trainy.ai/pluto). To take an interactive look, [try out our demo environment](https://demo.pluto.trainy.ai/o/dev-org) or [get an account with us today](https://pluto.trainy.ai/auth/sign-up)!

## See it in action

https://github.com/user-attachments/assets/6aff6448-00b6-41f2-adf4-4b7aa853ede6

## 🚀 Getting Started

Install the `pluto-ml` sdk

```bash
pip install -Uq "pluto-ml[full]"
```

```python
import pluto

pluto.init(project="hello-world")
pluto.log({"e": 2.718})
pluto.finish()
```

- Self-host your very own **Pluto** instance using the [Pluto Server](https://github.com/Trainy-ai/pluto-server) & get started in just 3 commands with **docker-compose**

```bash
git clone --recurse-submodules https://github.com/Trainy-ai/pluto-server.git; cd pluto-server
cp .env.example .env
sudo docker-compose --env-file .env up --build
```

You may also learn more about **Pluto** by checking out our [documentation](https://docs.trainy.ai/pluto).

<!-- You can try everything out in our [introductory tutorial](https://colab.research.google.com/github/Trainy-ai/pluto/blob/main/examples/intro.ipynb) and [torch tutorial](https://colab.research.google.com/github/Trainy-ai/pluto/blob/main/examples/torch.ipynb). -->

## Migration

### Neptune

Want to move your run data from Neptune to Pluto. Checkout the official docs from the Neptune transition hub [here](https://docs.neptune.ai/transition_hub/migration/to_pluto).

Before committing to Pluto, you want to see if there's parity between your Neptune and Pluto views? See our compatibility module documented [here](https://docs.trainy.ai/pluto/neptune-migration). Log to both Neptune and Pluto with a single import statement and no code changes.

## 🛠️ Development Setup

Want to contribute? Here's the quickest way to get the local toolchain (including the linters used in CI) running:

```bash
git clone https://github.com/Trainy-ai/pluto.git
cd pluto
python -m venv .venv && source .venv/bin/activate   # or use your preferred environment manager
python -m pip install --upgrade pip
pip install -e ".[full]"
```

Linting commands (mirrors `.github/workflows/lint.yml`):

```bash
bash format.sh
```

Run these locally before sending a PR to match the automation that checks on every push and pull request.

## 🫡 Vision

**Pluto** is a platform built for and by ML engineers, supported by [our community](https://discord.com/invite/HQUBJSVgAP)! We were tired of the current state of the art in ML observability tools, and this tool was born to help mitigate the inefficiencies - specifically, we hope to better inform you about your model performance and training runs; and actually **save you**, instead of charging you, for your precious compute time!

🌟 Be sure to star our repos if they help you ~
