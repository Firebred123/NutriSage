# NutriSage

A RAG (Retrieval-Augmented Generation) application that generates personalized health-focused recipes based on your health concerns.

## What it does

1. You describe a health concern (e.g., "I have trouble sleeping")
2. The system classifies it into a health category using AI
3. It retrieves beneficial ingredients from a database of 260+ items
4. A local LLM generates a personalized recipe using those ingredients

## Requirements

- Python 3.8+
- Ollama installed ([ollama.ai](https://ollama.ai))
- ~6GB RAM

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Pull the Mistral model for Ollama
ollama pull mistral
```

## Running the Application

### Option 1: Web Interface

```bash
# Start Ollama (in a separate terminal)
ollama serve

# Run the web server
python app.py
```

Open http://localhost:5000 in your browser.

### Option 2: Command Line

```bash
# Start Ollama (in a separate terminal)
ollama serve

# Run interactive CLI
python nutrisage_rag.py
```

## Project Files

| File | Description |
|------|-------------|
| `nutrisage_rag.py` | Core RAG system |
| `app.py` | Flask web server |
| `nutrisage_web.html` | Web interface |
| `nutri_sage_ingredients.csv` | Ingredient database |
| `requirements.txt` | Python dependencies |

## Example

```
You: I feel tired and have no energy

NutriSage:
- Classification: Low Energy and Fatigue (85% confidence)
- Ingredients: Spinach, Banana, Oatmeal, Eggs...
- Recipe: Energizing Morning Smoothie Bowl...
```

## Authors

- Paarth Goyal (2022343)
- Tarush Garg (2022537)
