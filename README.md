# Hackathon Project

A fast and efficient Python project setup using [uv](https://github.com/astral-sh/uv) for dependency management.

## Quick Start

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) installed

### Install uv

If you don't have `uv` installed, you can install it using:

**On Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**On macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or visit [uv Installation Guide](https://docs.astral.sh/uv/getting-started/installation/) for more options.

## Project Setup

### 1. Clone or Navigate to Project Directory

```bash
cd Hackaton
```

### 2. Create Virtual Environment and Install Dependencies

Using `uv` automatically creates and manages a virtual environment:

```bash
uv sync
```

This command will:
- Create a `.venv` virtual environment
- Install all dependencies from `pyproject.toml`
- Generate a `uv.lock` file for reproducible installs

### 3. Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

## Managing Dependencies

### Add a Package

To add a new package to the project:

```bash
uv add package_name
```

**Example:** Adding `flask` and `requests`
```bash
uv add flask requests
```

### Add a Development Package

For packages only needed during development (testing, linting, etc.):

```bash
uv add --dev package_name
```

**Example:** Adding pytest and black for development
```bash
uv add --dev pytest black
```

### Remove a Package

To remove a package from the project:

```bash
uv remove package_name
```

**Example:** Removing a package
```bash
uv remove old_package
```

### Update All Packages

To update all dependencies to their latest compatible versions:

```bash
uv sync --upgrade
```

### Update Specific Package

To update a specific package:

```bash
uv add --upgrade package_name
```

### View Installed Packages

To see all installed packages:

```bash
pip list
```

Or with uv:

```bash
uv pip list
```

## Environment Variables

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your configuration

3. The application will automatically load variables from `.env`

## Running the Application

Once the environment is set up:

```bash
python -m app
```

Or run specific scripts:

```bash
python path/to/script.py
```

## Project Structure

```
Hackaton/
├── app/                  # Main application code
├── data/                 # Data files and datasets
├── pyproject.toml        # Project metadata and dependencies
├── uv.lock              # Locked dependency versions (auto-generated)
├── .env.example         # Environment variables template
├── .env                 # Environment variables (local, not in git)
├── Dockerfile           # Container configuration
└── README.md            # This file
```

## Docker Support

Build the Docker image:

```bash
docker build -t hackathon-app .
```

Run the container:

```bash
docker run -it hackathon-app
```

## Tips for Hackathon Teams

- **Share Setup Quickly:** Everyone just needs to run `uv sync` once to get started
- **Consistent Versions:** The `uv.lock` file ensures everyone uses the same package versions
- **Fast Installation:** `uv` is significantly faster than pip for dependency resolution
- **Easy Collaboration:** When adding packages, just run `uv add` and commit `uv.lock` to git
- **Don't commit `.venv`:** The `.gitignore` already excludes it; everyone rebuilds with `uv sync`

## Common Commands Cheat Sheet

| Command | Purpose |
|---------|---------|
| `uv sync` | Install all dependencies |
| `uv add pkg` | Add a package |
| `uv add --dev pkg` | Add development package |
| `uv remove pkg` | Remove a package |
| `uv sync --upgrade` | Update all packages |
| `source .venv/bin/activate` | Activate virtual environment (macOS/Linux) |
| `.\.venv\Scripts\Activate.ps1` | Activate virtual environment (Windows PowerShell) |

## Troubleshooting

**Command not found: `uv`**
- Make sure uv is installed and in your PATH
- Try restarting your terminal after installation

**Virtual environment issues**
- Delete `.venv` folder and run `uv sync` again
- On Windows PowerShell, if activation fails, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Lock file conflicts**
- If you have merge conflicts in `uv.lock`, keep the version and run `uv sync` to regenerate