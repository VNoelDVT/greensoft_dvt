# greensoft_dvt

GreenSoft DVT is a prototype framework for analyzing and optimizing source code with a focus on **sustainability**.  
It combines heuristic analysis, AST parsing, machine learning, and LLM-powered recommendations (via NVIDIA NIM).

## 🚀 Installation
```
pip install -r requirements.txt
python example_7b_greensoft.py
```

This will:

Analyze a sample code snippet with multiple methods (Regex, AST, RandomForest, SBERT/Embeddings).

Query the NVIDIA NIM LLM (“GreenCoder”) to suggest more sustainable code.

Print results and, if enabled, plot evaluation charts.

📦 Requirements
Python 3.11 or 3.12 recommended

Dependencies are listed in requirements.txt (OpenAI client, NumPy, scikit-learn, matplotlib, sentence-transformers, etc.)

⚡ Notes
To use NVIDIA NIM models, set your API key as an environment variable:
```
export NVIDIA_API_KEY=your_key_here
```
```
$env:NVIDIA_API_KEY="your_key_here"
```

SBERT (Method 5) requires PyTorch. If you don’t plan to use it, you can remove torch and sentence-transformers from requirements.txt.

