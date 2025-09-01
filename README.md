# Improving Obscurity and Robustness in Unelicitable Backdoors

## Setup

On Linux/MacOS

### Step 1: Clone the repository

If you have not cloned the repository, clone it using

```sh
git clone git@github.com:bell-bot/msc-project.git
```

### Step 2: Create a virtual environment
Install the `python3.X-venv` package (substitute X for your specific Python 3 version, e.g 12), if it is not installed yet:

```sh
sudo apt install python3.X-venv
```

Inside the project's root directory, run

```sh
python -m venv .venv
source ./.venv/bin/activate
```

### Step 3: Install the dependencies

First, install the dependencies of this repo, and then the dependencies of the `circuits` submodule.

```sh
pip install -e .
cd circuits
pip install -e .
cd ../
```





