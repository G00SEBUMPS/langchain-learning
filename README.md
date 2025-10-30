# LangChain Learning Project

This repository contains my learning journey and experiments with LangChain, a framework for developing applications powered by language models.

## Project Overview

This project is focused on learning and implementing various features of LangChain, including:
- Chains
- Agents
- Memory
- Prompts
- Document loading and manipulation
- Vector stores and embeddings

## Setup

1. Clone the repository:
```bash
git clone https://github.com/G00SEBUMPS/langchain-learning.git
# LangChain Learning Project

This repository contains my learning experiments with LangChain, a framework for building applications powered by language models.

## Project Overview

This project is focused on learning and implementing various features of LangChain, including:
- Chains
- Agents
- Memory
- Prompts
- Document loading and manipulation
- Vector stores and embeddings

## Setup

1. Clone the repository:
```bash
git clone https://github.com/G00SEBUMPS/langchain-learning.git
cd langchain-learning
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies (adjust as your project grows):
```bash
pip install -r requirements.txt
```

## Using Ollama locally

If you run an Ollama server locally (default listen port 11434), the `main.py` script will try to detect it automatically and configure the Ollama client to use it. You can also force the code to use a specific Ollama base URL by setting the `OLLAMA_URL` environment variable.

Examples:

Start Ollama (if you have it installed) and run a model locally. See Ollama docs for exact commands. By default Ollama serves HTTP on port 11434.

Set the environment variable to point at a custom host/port:
```bash
export OLLAMA_URL="http://localhost:11434"
python main.py
```

If the script detects a running Ollama at `http://localhost:11434`, it will print a message and set `OLLAMA_URL` for the running process so the client libraries can pick it up.

## Project Structure

- `main.py`: Main application entry point (contains Ollama detection logic)
- `pyproject.toml`: Project dependencies and configuration
- `.gitignore`: Files Git should ignore
- `README.md`: Project documentation

## Usage

Run the example in `main.py` after activating your virtual environment and ensuring any necessary dependencies are installed. The script currently demonstrates using `ChatOllama` from `langchain_ollama`.

## Contributing

Feel free to fork this repository and submit pull requests for improvements.

## License

MIT

## Contact

GitHub: [@G00SEBUMPS](https://github.com/G00SEBUMPS)