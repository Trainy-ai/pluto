[![pypi](https://img.shields.io/pypi/v/mlop)](https://pypi.org/project/trainy-mlop-nightly/)

## THIS README/REPO IS CURRENTLY UNDER CONSTRUCTION WHILE WE UPDATE THE REFERENCES IN OUR FORK

**mlop** is a Machine Learning Operations (MLOps) framework. It provides [self-hostable superior experimental tracking capabilities and lifecycle management for training ML models](https://github.com/mlop-ai/server). To get started, [try out our introductory notebook](https://colab.research.google.com/github/mlop-ai/mlop/blob/main/examples/intro.ipynb) or [get an account with us today](https://trakkur.trainy.ai/auth/sign-up)!

## 🎥 Demo

**mlop** adopts a KISS philosophy that allows it to outperform all other tools in this category. Supporting high and stable data throughput should be *THE* top priority for efficient MLOps.
<video loop src='https://github.com/user-attachments/assets/efd9720e-6128-4278-85ec-ee6139a851af' alt="demo" width="1200" style="display: block; margin: auto;"></video>

<p align="center">
<strong>mlop</strong> logger (bottom left) v. a conventional logger (bottom right)
</p>

## 🚀 Getting Started

- Try **mlop** on our platform in [a notebook](https://colab.research.google.com/github/mlop-ai/mlop/blob/main/examples/intro.ipynb) & start integrating in just 5 lines of Python code:

```python
%pip install -Uq "mlop[full]"
import mlop

mlop.init(project="hello-world")
mlop.log({"e": 2.718})
mlop.finish()
```

- Self-host your very own **mlop** instance & get started in just 3 commands with **docker-compose**

```bash
git clone --recurse-submodules https://github.com/mlop-ai/server.git; cd server
cp .env.example .env
sudo docker-compose --env-file .env up --build
```

You may also learn more about **mlop** by checking out our [documentation](https://docs.mlop.ai/).

You can try everything out in our [introductory tutorial](https://colab.research.google.com/github/mlop-ai/mlop/blob/main/examples/intro.ipynb) and [torch tutorial](https://colab.research.google.com/github/mlop-ai/mlop/blob/main/examples/torch.ipynb).  

## 🛠️ Development Setup

Want to contribute? Here's the quickest way to get the local toolchain (including the linters used in CI) running:

```bash
git clone https://github.com/mlop-ai/mlop.git
cd mlop
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

**mlop** is a platform built for and by ML engineers, supported by [our community](https://discord.com/invite/HQUBJSVgAP)! We were tired of the current state of the art in ML observability tools, and this tool was born to help mitigate the inefficiencies - specifically, we hope to better inform you about your model performance and training runs; and actually **save you**, instead of charging you, for your precious compute time! 

🌟 Be sure to star our repos if they help you ~
