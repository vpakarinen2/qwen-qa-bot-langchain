# Command-line question-answering (Q/A) bot.

## Setup

1. **Clone the repository**
- ``git clone https://github.com/vpakarinen2/qa-bot-langchain.git``
- ``cd qa-bot-langchain``

2. **Create virtual environment (Python 3.11)**
- ``python -m venv .venv``
- ``.venv\Scripts\Activate.ps1`` (source .venv/bin/activate on Linux)

3. **Install dependencies**

- ``python -m pip install --upgrade pip``
- ``pip install -r requirements.txt``

## Usage
1. **Run the Python script**

- ``python qa_bot.py``

## CLI arguments

### Options

- `-q`, `--question`  
  - Description: The question to ask the model.

- `-m`, `--model-name`  
  - Description: Hugging Face model id (e.g. Qwen/Qwen3-4B-Thinking-2507).
 
- `-l`, `--lora-name`  
  - Description: Optional LoRA adapter to apply on top of the base model.
 
- `-t`, `--tool`  
  - Description: Optional tool to use (currently: math, time).
 
- `-c`, `--city`  
  - Description: City for time tool (default: Helsinki).
 
- `-p`, `--prompt-style`  
  - Description: Prompt template style to use.
 
- `-n`, `--max-new-tokens`  
  - Description: Maximum number of new tokens to generate.
 
- `-r`, `--raw-output`  
  - Description: Print raw model output (disables trimming).

- `--trust-remote-code`
  - Description: Allow execution of custom remote code.

 ## Example Output

 ```
  Model:    Qwen/Qwen3-4B-Thinking-2507
  LoRA:     None
  Question: A bat and ball cost 1.10 USD total. The bat costs 1 USD more than the ball. How much does the ball cost?
  Answer:   The ball costs 5 cents.
```

## Author

Ville Pakarinen (@vpakarinen2)
