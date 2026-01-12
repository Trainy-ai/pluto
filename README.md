[![pypi](https://img.shields.io/pypi/v/pluto-ml)](https://pypi.org/project/pluto-ml/)

## THIS README/REPO IS CURRENTLY UNDER CONSTRUCTION WHILE WE UPDATE THE REFERENCES IN OUR FORK

**Pluto** is a Machine Learning Operations (MLOps) framework. It provides [self-hostable superior experimental tracking capabilities and lifecycle management for training ML models](https://docs.trainy.ai/pluto). To get started, [try out our introductory notebook](https://colab.research.google.com/github/Trainy-ai/pluto/blob/main/examples/intro.ipynb) or [get an account with us today](https://pluto.trainy.ai/auth/sign-up)!

## 🎥 Demo

**Pluto** adopts a KISS philosophy that allows it to outperform all other tools in this category. Supporting high and stable data throughput should be *THE* top priority for efficient MLOps.
<video loop src='https://github.com/user-attachments/assets/efd9720e-6128-4278-85ec-ee6139a851af' alt="demo" width="1200" style="display: block; margin: auto;"></video>

<p align="center">
<strong>Pluto</strong> logger (bottom left) v. a conventional logger (bottom right)
</p>

## 🚀 Getting Started

- Try **Pluto** on our platform in [a notebook](https://colab.research.google.com/github/Trainy-ai/pluto/blob/main/examples/intro.ipynb) & start integrating in just 5 lines of Python code:

```python
%pip install -Uq "pluto-ml[full]"
import pluto

pluto.init(project="hello-world")
pluto.log({"e": 2.718})
pluto.finish()
```

- Self-host your very own **Pluto** instance & get started in just 3 commands with **docker-compose**

```bash
git clone --recurse-submodules https://github.com/mlop-ai/server.git; cd server
cp .env.example .env
sudo docker-compose --env-file .env up --build
```

You may also learn more about **Pluto** by checking out our [documentation](https://docs.trainy.ai/pluto).

<!-- You can try everything out in our [introductory tutorial](https://colab.research.google.com/github/Trainy-ai/pluto/blob/main/examples/intro.ipynb) and [torch tutorial](https://colab.research.google.com/github/Trainy-ai/pluto/blob/main/examples/torch.ipynb). -->

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
