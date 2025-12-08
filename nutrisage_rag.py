"""
NutriSage RAG Application
=========================
A Retrieval-Augmented Generation system that:
1. Uses transformer-based zero-shot classification to map user queries to ailment categories
2. Retrieves relevant ingredients from the database
3. Uses a local LLM (via Ollama) to generate helpful recipes

Requirements:
- Python 3.8+
- transformers library (Hugging Face)
- torch (PyTorch)
- Ollama installed locally (https://ollama.ai)
- A 7B parameter model pulled (e.g., mistral, llama2, gemma)

First-time setup will download the classification model (~1.5GB)

Usage:
    python nutrisage_rag.py
"""

import json
import csv
import requests
import re
import os
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# Transformers for zero-shot classification
from transformers import pipeline
import torch

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Ingredient:
    """Represents an ingredient with its health properties"""
    id: int
    name: str
    ailment_category: str
    specific_ailments: List[str]
    positive_effects: List[str]
    negative_effects: List[str]
    recipe_tags: List[str]
    serving_suggestion: str
    
    def to_context_string(self) -> str:
        """Convert ingredient to a string for LLM context"""
        return (
            f"- {self.name}: "
            f"Benefits: {', '.join(self.positive_effects[:3])}. "
            f"Best used: {self.serving_suggestion}. "
            f"Caution: {self.negative_effects[0] if self.negative_effects else 'None'}."
        )
    
    def to_detailed_string(self) -> str:
        """Detailed ingredient information"""
        return (
            f"Ingredient: {self.name}\n"
            f"  Category: {self.ailment_category}\n"
            f"  Helps with: {', '.join(self.specific_ailments)}\n"
            f"  Benefits: {', '.join(self.positive_effects)}\n"
            f"  Cautions: {', '.join(self.negative_effects) if self.negative_effects else 'None'}\n"
            f"  Serving: {self.serving_suggestion}"
        )

@dataclass
class ClassificationResult:
    """Result of zero-shot classification"""
    category: str
    confidence: float
    all_scores: Dict[str, float] = field(default_factory=dict)

@dataclass
class RAGResponse:
    """Complete response from the RAG system"""
    user_query: str
    classified_category: str
    confidence: float
    retrieved_ingredients: List[Ingredient]
    generated_recipe: str
    all_classifications: Dict[str, float] = field(default_factory=dict)
    health_disclaimer: str = (
        "Disclaimer: This recipe is for informational purposes only. "
        "Please consult a healthcare professional before making dietary changes "
        "for medical conditions."
    )

# =============================================================================
# TRANSFORMER-BASED ZERO-SHOT CLASSIFIER
# =============================================================================

class ZeroShotAilmentClassifier:
    """
    Zero-shot classification using transformer models (e.g., BART-large-MNLI).
    
    This classifier uses a pre-trained NLI model to classify health queries
    into ailment categories WITHOUT any training on health data.
    
    Supported models:
    - facebook/bart-large-mnli (default, best accuracy)
    - MoritzLaworz/bart-large-mnli-distilled-s2 (faster, slightly less accurate)
    - valhalla/distilbart-mnli-12-1 (smallest, fastest)
    - cross-encoder/nli-deberta-v3-small (good balance)
    """
    
    def __init__(self, 
                 model_name: str = "facebook/bart-large-mnli",
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize the zero-shot classifier.
        
        Args:
            model_name: Hugging Face model name for zero-shot classification
            device: Device to run on ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache the model (default: ~/.cache/huggingface)
        """
        print(f"  Loading zero-shot classification model: {model_name}")
        print(f"  (This may take a moment on first run as the model downloads...)")
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        
        # Initialize the pipeline
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if device == "cuda" else -1,
            model_kwargs={"cache_dir": cache_dir} if cache_dir else {}
        )
        
        # Define ailment categories and their mappings
        self.categories = self._get_categories()
        self.category_labels = list(self.categories.keys())
        
        print(f"  Model loaded on {device.upper()}")
        print(f"  {len(self.category_labels)} health categories configured")
    
    def _get_categories(self) -> Dict:
        """
        Define ailment categories with their database mappings.
        The keys are human-readable category names used for classification.
        """
        return {
            "Digestive and Stomach Issues": {
                "db_categories": ["Digestive Health", "Digestive Issues"],
                "description": "Digestive system issues including stomach problems, bloating, acid reflux, and gut health"
            },
            "Heart and Cardiovascular Health": {
                "db_categories": ["Heart Health", "Mood and Heart", "Immune and Heart", "Blood Pressure"],
                "description": "Cardiovascular health, cholesterol, blood pressure, and heart-related conditions"
            },
            "Joint Pain and Inflammation": {
                "db_categories": ["Inflammation", "Joint Health"],
                "description": "Inflammatory conditions, arthritis, joint pain, and muscle soreness"
            },
            "Immune System and Infections": {
                "db_categories": ["Immune Support", "Energy and Immunity"],
                "description": "Immune system support, cold, flu, and infection prevention"
            },
            "Sleep Problems and Insomnia": {
                "db_categories": ["Sleep Support", "Sleep Issues"],
                "description": "Sleep disorders, insomnia, restlessness, and sleep quality improvement"
            },
            "Brain Health and Cognitive Function": {
                "db_categories": ["Cognitive Health", "Brain Health", "Focus and Energy"],
                "description": "Memory, focus, concentration, brain fog, and cognitive function"
            },
            "Low Energy and Fatigue": {
                "db_categories": ["Energy Support", "Energy and Muscles", "Energy and Brain", 
                                "Energy and Nutrition", "Energy and Mood"],
                "description": "Fatigue, tiredness, low energy, and stamina issues"
            },
            "Stress and Anxiety": {
                "db_categories": ["Stress Relief", "Stress and Energy"],
                "description": "Stress management, anxiety, nervousness, and relaxation"
            },
            "Blood Sugar and Diabetes": {
                "db_categories": ["Blood Sugar"],
                "description": "Blood sugar regulation, diabetes management, and glucose control"
            },
            "Anemia and Blood Health": {
                "db_categories": ["Blood Building", "Blood Health"],
                "description": "Anemia, iron deficiency, and blood-related conditions"
            },
            "Bone Health and Osteoporosis": {
                "db_categories": ["Bone Health"],
                "description": "Bone density, osteoporosis, calcium, and skeletal health"
            },
            "Skin Problems and Dermatology": {
                "db_categories": ["Skin Health", "Skin and Energy"],
                "description": "Skin conditions, acne, eczema, aging skin, and complexion"
            },
            "Respiratory and Breathing Issues": {
                "db_categories": ["Respiratory Health", "Respiratory Issues"],
                "description": "Breathing problems, cough, asthma, congestion, and lung health"
            },
            "Eye Health and Vision": {
                "db_categories": ["Eye Health"],
                "description": "Vision problems, eye strain, and eye health"
            },
            "Liver Health and Detoxification": {
                "db_categories": ["Liver Health", "Detox Support"],
                "description": "Liver function, detoxification, and toxin removal"
            },
            "Kidney and Urinary Health": {
                "db_categories": ["Kidney Health", "Urinary Health"],
                "description": "Kidney function, UTI, bladder health, and urinary issues"
            },
            "Hormonal Imbalance": {
                "db_categories": ["Hormone Balance", "Thyroid Health"],
                "description": "Hormonal balance, menopause, thyroid, and endocrine health"
            },
            "Weight Management and Metabolism": {
                "db_categories": ["Weight Management", "Metabolism"],
                "description": "Weight loss, obesity, metabolism, and appetite control"
            },
            "Muscle Health and Recovery": {
                "db_categories": ["Muscle Health", "Energy and Muscles"],
                "description": "Muscle building, cramps, recovery, and athletic performance"
            },
            "Mood and Emotional Wellbeing": {
                "db_categories": ["Mood Support", "Energy and Mood"],
                "description": "Depression, mood swings, emotional balance, and happiness"
            }
        }
    
    def classify(self, 
                 user_input: str, 
                 top_k: int = 3,
                 multi_label: bool = False) -> List[ClassificationResult]:
        """
        Classify user input into ailment categories using zero-shot classification.
        
        Args:
            user_input: Natural language description of symptoms/health concerns
            top_k: Number of top categories to return
            multi_label: If True, allows multiple labels to be predicted independently
            
        Returns:
            List of ClassificationResult sorted by confidence
        """
        if not user_input.strip():
            return []
        
        # Run zero-shot classification
        result = self.classifier(
            user_input,
            candidate_labels=self.category_labels,
            multi_label=multi_label
        )
        
        # Parse results
        all_scores = dict(zip(result['labels'], result['scores']))
        
        # Create sorted results
        classifications = []
        for label, score in zip(result['labels'][:top_k], result['scores'][:top_k]):
            classifications.append(ClassificationResult(
                category=label,
                confidence=round(score, 4),
                all_scores=all_scores
            ))
        
        return classifications
    
    def get_db_categories(self, category: str) -> List[str]:
        """Get database category names for a classified category"""
        if category in self.categories:
            return self.categories[category].get("db_categories", [])
        return []
    
    def get_all_labels(self) -> List[str]:
        """Get all available category labels"""
        return self.category_labels.copy()

# =============================================================================
# INGREDIENT DATABASE (RETRIEVAL COMPONENT)
# =============================================================================

class IngredientDatabase:
    """
    Manages the ingredient database for retrieval.
    Acts as the 'R' in RAG.
    """
    
    def __init__(self, csv_path: str = "nutri_sage_ingredients.csv"):
        """Load and index ingredients from CSV"""
        self.ingredients: List[Ingredient] = []
        self.category_index: Dict[str, List[Ingredient]] = defaultdict(list)
        self.tag_index: Dict[str, List[Ingredient]] = defaultdict(list)
        
        self._load_csv(csv_path)
        self._build_indices()
    
    def _load_csv(self, csv_path: str):
        """Load ingredients from CSV file"""
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ingredient = Ingredient(
                    id=int(row['id']),
                    name=row['ingredient_name'],
                    ailment_category=row['ailment_category'],
                    specific_ailments=[s.strip() for s in row['specific_ailments'].split(',')],
                    positive_effects=[e.strip() for e in row['positive_effects'].split(',')],
                    negative_effects=[e.strip() for e in row['negative_effects'].split(',') if e.strip()],
                    recipe_tags=[t.strip() for t in row['recipe_tags'].split(';')],
                    serving_suggestion=row['serving_suggestion']
                )
                self.ingredients.append(ingredient)
    
    def _build_indices(self):
        """Build indices for fast retrieval"""
        for ing in self.ingredients:
            # Index by category
            self.category_index[ing.ailment_category].append(ing)
            
            # Index by tags
            for tag in ing.recipe_tags:
                self.tag_index[tag.lower()].append(ing)
    
    def retrieve_by_categories(self, categories: List[str], limit: int = 10) -> List[Ingredient]:
        """Retrieve ingredients matching any of the given categories"""
        results = []
        seen_names = set()
        
        for category in categories:
            for ing in self.category_index.get(category, []):
                if ing.name not in seen_names:
                    results.append(ing)
                    seen_names.add(ing.name)
        
        # Randomize the order of ingredients found
        random.shuffle(results) 
        
        return results[:limit]
    
    def retrieve_by_ailment(self, ailment: str, limit: int = 10) -> List[Ingredient]:
        """Retrieve ingredients that help with a specific ailment"""
        ailment_lower = ailment.lower()
        results = []
        
        for ing in self.ingredients:
            for specific in ing.specific_ailments:
                if ailment_lower in specific.lower():
                    results.append(ing)
                    break
        
        return results[:limit]
    
    def retrieve_by_tags(self, tags: List[str], limit: int = 10) -> List[Ingredient]:
        """Retrieve ingredients matching recipe tags"""
        results = []
        seen_names = set()
        
        for tag in tags:
            for ing in self.tag_index.get(tag.lower(), []):
                if ing.name not in seen_names:
                    results.append(ing)
                    seen_names.add(ing.name)
        
        return results[:limit]
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            "total_ingredients": len(self.ingredients),
            "categories": len(self.category_index),
            "category_counts": {cat: len(ings) for cat, ings in self.category_index.items()}
        }

# =============================================================================
# OLLAMA LLM CLIENT (GENERATION COMPONENT)
# =============================================================================

class OllamaClient:
    """
    Client for interacting with Ollama local LLM.
    Acts as the 'G' in RAG.
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model: str = "mistral",
                 timeout: int = 120):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL
            model: Model name (mistral, llama2, gemma, etc.)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
    
    def is_available(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except:
            pass
        return []
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for context
            
        Returns:
            Generated text
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 1024
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Error: Ollama returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Please ensure Ollama is running (ollama serve)."
        except requests.exceptions.Timeout:
            return "Error: Request timed out. The model may be loading or the prompt is too complex."
        except Exception as e:
            return f"Error: {str(e)}"

# =============================================================================
# RAG RECIPE GENERATOR
# =============================================================================

class NutriSageRAG:
    """
    Main RAG application combining:
    - Zero-shot classification (transformer-based) for query understanding
    - Ingredient retrieval from database
    - LLM-based recipe generation via Ollama
    """
    
    def __init__(self, 
                 csv_path: str = "nutri_sage_ingredients.csv",
                 ollama_model: str = "mistral",
                 classifier_model: str = "facebook/bart-large-mnli",
                 device: Optional[str] = None):
        """
        Initialize NutriSage RAG system.
        
        Args:
            csv_path: Path to ingredients CSV
            ollama_model: Ollama model to use for recipe generation
            classifier_model: Hugging Face model for zero-shot classification
            device: Device for classification model ('cuda', 'cpu', or None for auto)
        """
        print("Initializing NutriSage RAG System...")
        print("=" * 60)
        
        # Initialize zero-shot classifier
        self.classifier = ZeroShotAilmentClassifier(
            model_name=classifier_model,
            device=device
        )
        
        # Initialize ingredient database
        print(f"\n  Loading ingredient database...")
        self.database = IngredientDatabase(csv_path)
        print(f"  Database loaded ({self.database.get_stats()['total_ingredients']} ingredients)")
        
        # Initialize Ollama client
        print(f"\n  Connecting to Ollama...")
        self.llm = OllamaClient(model=ollama_model)
        
        if self.llm.is_available():
            print(f"  Ollama connected")
            print(f"  Using model: {ollama_model}")
        else:
            print("  Warning: Ollama server is not reachable at localhost:11434")
        
        print("\n" + "=" * 60)
        print("NutriSage ready!\n")
    
    def _build_recipe_prompt(self, 
                            query: str, 
                            category: str, 
                            ingredients: List[Ingredient]) -> str:
        """Build the prompt for recipe generation"""
        
        # Format ingredients for context
        ingredient_context = "\n".join([ing.to_context_string() for ing in ingredients])
        
        prompt = f"""Based on the user's health concern and the provided beneficial ingredients, create a helpful recipe.

USER'S HEALTH CONCERN: {query}
IDENTIFIED CATEGORY: {category}

BENEFICIAL INGREDIENTS FOR THIS CONDITION:
{ingredient_context}

TASK: Create ONE complete, practical recipe using 3-5 of these beneficial ingredients. The recipe should:
1. Be easy to prepare (under 30 minutes)
2. Taste good while being healthy
3. Maximize the health benefits for the user's condition
4. Include clear instructions

FORMAT YOUR RESPONSE AS:
RECIPE NAME: [Creative name]

INGREDIENTS:
- [List each ingredient with quantity]

INSTRUCTIONS:
1. [Step-by-step instructions]

HEALTH BENEFITS:
[Brief explanation of how this recipe helps with {category}]

PREP TIME: [X minutes]
SERVINGS: [X servings]

Generate the recipe now:"""
        
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM"""
        return """You are NutriSage, an expert culinary health assistant. Your role is to create delicious, 
practical recipes that incorporate ingredients known to help with specific health conditions.

Guidelines:
- Create recipes that are both tasty AND healthy
- Use simple, accessible ingredients when possible
- Provide clear, easy-to-follow instructions
- Explain the health benefits of key ingredients
- Be encouraging and supportive
- Always remind users to consult healthcare professionals for medical advice

You specialize in creating healing recipes that make healthy eating enjoyable."""
    
    def process_query(self, user_query: str, num_ingredients: int = 8) -> RAGResponse:
        """
        Process a user health query through the full RAG pipeline.
        
        Args:
            user_query: User's health concern in natural language
            num_ingredients: Number of ingredients to retrieve
            
        Returns:
            RAGResponse with classification, ingredients, and generated recipe
        """
        # Step 1: Zero-shot Classification using transformer model
        print(f"\n  Classifying health concern...")
        classifications = self.classifier.classify(user_query, top_k=3)
        
        if not classifications:
            return RAGResponse(
                user_query=user_query,
                classified_category="Unknown",
                confidence=0.0,
                retrieved_ingredients=[],
                generated_recipe="I couldn't identify a specific health concern from your query. "
                               "Could you please describe your symptoms more specifically? "
                               "For example: 'I have trouble sleeping' or 'My stomach feels upset'"
            )
        
        primary_classification = classifications[0]
        category = primary_classification.category
        confidence = primary_classification.confidence
        
        print(f"  Classified as: {category} ({confidence:.1%} confidence)")
        
        # Step 2: Retrieve relevant ingredients
        print(f"  Retrieving beneficial ingredients...")
        db_categories = self.classifier.get_db_categories(category)
        random_limit = random.randint(4, 10)
        print(f"  Selection: Picking {random_limit} random ingredients...")
        
        ingredients = self.database.retrieve_by_categories(db_categories, limit=random_limit)
        
        # If not enough ingredients, try secondary classification
        if len(ingredients) < 5 and len(classifications) > 1:
            secondary_db_cats = self.classifier.get_db_categories(classifications[1].category)
            additional = self.database.retrieve_by_categories(secondary_db_cats, limit=5)
            for ing in additional:
                if ing.name not in [i.name for i in ingredients]:
                    ingredients.append(ing)
                    if len(ingredients) >= num_ingredients:
                        break
        
        print(f"  Retrieved {len(ingredients)} ingredients")
        
        # Step 3: Generate recipe using LLM
        print(f"  Generating personalized recipe...")
        prompt = self._build_recipe_prompt(user_query, category, ingredients)
        system_prompt = self._get_system_prompt()
        
        generated_recipe = self.llm.generate(prompt, system_prompt)
        
        # Handle LLM errors
        if generated_recipe.startswith("Error:"):
            print(f"  LLM unavailable, using fallback recipe")
            generated_recipe = self._fallback_recipe(category, ingredients)
        else:
            print(f"  Recipe generated!")
        
        # Build all classifications dict
        all_classifications = {c.category: c.confidence for c in classifications}
        
        return RAGResponse(
            user_query=user_query,
            classified_category=category,
            confidence=confidence,
            retrieved_ingredients=ingredients,
            generated_recipe=generated_recipe,
            all_classifications=all_classifications
        )
    
    def _fallback_recipe(self, category: str, ingredients: List[Ingredient]) -> str:
        """Generate a simple fallback recipe when LLM is unavailable"""
        if not ingredients:
            return "No suitable ingredients found for your condition."
        
        selected = ingredients[:4]
        
        recipe = f"""RECIPE: Healing {category.split(' and ')[0]} Bowl

INGREDIENTS:
"""
        for ing in selected:
            recipe += f"- {ing.name} ({ing.serving_suggestion})\n"
        
        recipe += f"""
INSTRUCTIONS:
1. Prepare each ingredient according to its serving suggestion
2. Combine in a bowl or plate
3. Season to taste with herbs and a squeeze of lemon
4. Enjoy mindfully

HEALTH BENEFITS:
This combination includes ingredients known to help with {category}:
"""
        for ing in selected[:3]:
            recipe += f"- {ing.name}: {', '.join(ing.positive_effects[:2])}\n"
        
        recipe += """
PREP TIME: 15-20 minutes
SERVINGS: 1-2 servings

Note: Ollama LLM is not available. This is a simplified recipe suggestion.
Start Ollama with 'ollama serve' for personalized recipe generation."""
        
        return recipe
    
    def interactive_session(self):
        """Run an interactive chat session"""
        print("=" * 60)
        print("Welcome to NutriSage - Your Culinary Health Assistant")
        print("=" * 60)
        print("\nTell me about your health concerns, and I'll create a")
        print("personalized recipe to help you feel better!")
        print("\nType 'quit' to exit, 'help' for examples\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("\nThank you for using NutriSage. Stay healthy!")
                    break
                
                if user_input.lower() == 'help':
                    self._print_help()
                    continue
                
                if user_input.lower() == 'stats':
                    self._print_stats()
                    continue
                
                if user_input.lower() == 'categories':
                    self._print_categories()
                    continue
                
                # Process the query
                response = self.process_query(user_input)
                
                # Display results
                self._display_response(response)
                
            except KeyboardInterrupt:
                print("\n\nSession ended. Stay healthy!")
                break
    
    def _display_response(self, response: RAGResponse):
        """Display the RAG response in a formatted way"""
        print("\n" + "-" * 60)
        print(f"CLASSIFICATION: {response.classified_category}")
        print(f"Confidence: {response.confidence:.1%}")
        
        # Show top 3 classifications
        if response.all_classifications:
            print("\nAll scores:")
            sorted_scores = sorted(response.all_classifications.items(), 
                                  key=lambda x: x[1], reverse=True)[:3]
            for cat, score in sorted_scores:
                bar = "#" * int(score * 20) + "." * (20 - int(score * 20))
                print(f"  [{bar}] {score:.1%} {cat[:35]}")
        
        print("-" * 60)
        
        if response.retrieved_ingredients:
            print("\nRETRIEVED INGREDIENTS:")
            for ing in response.retrieved_ingredients[:5]:
                benefits = ", ".join(ing.positive_effects[:2])
                print(f"  - {ing.name}: {benefits}")
        
        print("\n" + "-" * 60)
        print("GENERATED RECIPE:")
        print("-" * 60)
        print(response.generated_recipe)
        
        print("\n" + "-" * 60)
        print(f"Note: {response.health_disclaimer}")
        print("-" * 60)
    
    def _print_help(self):
        """Print help information"""
        print("""
============================================================
                    NutriSage Help                          
============================================================
EXAMPLE QUERIES:
  - "I have trouble sleeping at night"
  - "My stomach feels upset after eating"
  - "I feel tired and have no energy"
  - "I'm stressed and anxious about work"
  - "My joints are aching"
  - "I want to boost my immune system"
  - "I need to manage my blood sugar"
  - "I've been suffering from indigestion and acidity"

COMMANDS:
  quit       - Exit the application
  help       - Show this help message
  stats      - Show database statistics
  categories - Show all health categories
============================================================
""")
    
    def _print_stats(self):
        """Print database statistics"""
        stats = self.database.get_stats()
        print(f"""
============================================================
                  Database Statistics                       
============================================================
Total Ingredients: {stats['total_ingredients']}
Categories: {stats['categories']}
Classification Model: {self.classifier.model_name}
Device: {self.classifier.device.upper()}
------------------------------------------------------------
Top Database Categories:""")
        
        sorted_cats = sorted(stats['category_counts'].items(), key=lambda x: x[1], reverse=True)
        for cat, count in sorted_cats[:10]:
            print(f"  - {cat}: {count}")
        
        print("============================================================")
    
    def _print_categories(self):
        """Print all classification categories"""
        print("""
============================================================
              Health Classification Categories              
============================================================""")
        
        for i, label in enumerate(self.classifier.get_all_labels(), 1):
            print(f"  {i:2}. {label}")
        
        print("============================================================")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for NutriSage RAG application"""
    import sys
    
    # Default settings
    ollama_model = "mistral"
    classifier_model = "facebook/bart-large-mnli"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        ollama_model = sys.argv[1]
    if len(sys.argv) > 2:
        classifier_model = sys.argv[2]
    
    print("""
============================================================
         NutriSage RAG Application                    
                                                            
  Transformer-based Zero-Shot Classification                
  + Local LLM Recipe Generation via Ollama                  
============================================================
""")
    
    # Initialize and run
    try:
        rag = NutriSageRAG(
            csv_path="nutri_sage_ingredients.csv",
            ollama_model=ollama_model,
            classifier_model=classifier_model
        )
        rag.interactive_session()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the ingredients database file is in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing NutriSage: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()