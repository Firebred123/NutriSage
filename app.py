"""
NutriSage Web Server
====================
Flask-based web server that provides:
1. REST API for the RAG system with transformer-based zero-shot classification
2. Serves the web interface
3. Handles real-time recipe generation

Run with: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import sys

# Import our RAG system
from nutrisage_rag import NutriSageRAG

app = Flask(__name__)
CORS(app)

# Global RAG system instance
rag_system = None

def get_rag():
    """Lazy initialization of RAG system"""
    global rag_system
    if rag_system is None:
        # --- CHANGE: Hardcoded model name ---
        ollama_model = "hf.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF:Q4_K_M"
        
        # Keep classifier flexible or hardcode it too if you want
        classifier_model = "facebook/bart-large-mnli"
        
        rag_system = NutriSageRAG(
            csv_path="nutri_sage_ingredients.csv",
            ollama_model=ollama_model,
            classifier_model=classifier_model
        )
    return rag_system

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    rag = get_rag()
    ollama_status = "connected" if rag.llm.is_available() else "disconnected"
    
    return jsonify({
        'status': 'healthy',
        'service': 'NutriSage RAG API',
        'ollama': ollama_status,
        'ollama_model': rag.llm.model,
        'classifier_model': rag.classifier.model_name,
        'classifier_device': rag.classifier.device,
        'database_size': rag.database.get_stats()['total_ingredients']
    })

@app.route('/api/query', methods=['POST'])
def process_query():
    """
    Main RAG query endpoint.
    
    Request body:
    {
        "query": "I have trouble sleeping",
        "num_ingredients": 8  // optional
    }
    """
    try:
        data = request.json
        user_query = data.get('query', '').strip()
        num_ingredients = data.get('num_ingredients', 8)
        
        if not user_query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Process through RAG pipeline
        rag = get_rag()
        response = rag.process_query(user_query, num_ingredients)
        
        # Format response
        return jsonify({
            'success': True,
            'query': response.user_query,
            'classification': {
                'category': response.classified_category,
                'confidence': response.confidence,
                'all_scores': response.all_classifications
            },
            'ingredients': [
                {
                    'name': ing.name,
                    'category': ing.ailment_category,
                    'benefits': ing.positive_effects[:3],
                    'cautions': ing.negative_effects[:2] if ing.negative_effects else [],
                    'serving': ing.serving_suggestion
                }
                for ing in response.retrieved_ingredients
            ],
            'recipe': response.generated_recipe,
            'disclaimer': response.health_disclaimer
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def classify_only():
    """
    Classify a query without generating a recipe.
    Useful for quick category identification.
    """
    try:
        data = request.json
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({'error': 'No query provided'}), 400
        
        rag = get_rag()
        classifications = rag.classifier.classify(user_query, top_k=5)
        
        return jsonify({
            'success': True,
            'query': user_query,
            'classifications': [
                {
                    'category': c.category,
                    'confidence': c.confidence
                }
                for c in classifications
            ]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ingredients', methods=['GET'])
def get_ingredients():
    """Get ingredients for a specific category"""
    category = request.args.get('category', '')
    limit = int(request.args.get('limit', 10))
    
    rag = get_rag()
    
    if category:
        # Map display category to database categories
        db_cats = rag.classifier.get_db_categories(category)
        if not db_cats:
            db_cats = [category]
        ingredients = rag.database.retrieve_by_categories(db_cats, limit=limit)
    else:
        ingredients = rag.database.ingredients[:limit]
    
    return jsonify({
        'success': True,
        'count': len(ingredients),
        'ingredients': [
            {
                'id': ing.id,
                'name': ing.name,
                'category': ing.ailment_category,
                'benefits': ing.positive_effects,
                'cautions': ing.negative_effects,
                'serving': ing.serving_suggestion,
                'helps_with': ing.specific_ailments
            }
            for ing in ingredients
        ]
    })

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all available classification categories"""
    rag = get_rag()
    stats = rag.database.get_stats()
    
    categories = []
    for cat_name in rag.classifier.get_all_labels():
        cat_data = rag.classifier.categories.get(cat_name, {})
        db_cats = cat_data.get('db_categories', [])
        count = sum(stats['category_counts'].get(dc, 0) for dc in db_cats)
        
        categories.append({
            'name': cat_name,
            'description': cat_data.get('description', ''),
            'ingredient_count': count,
            'db_categories': db_cats
        })
    
    return jsonify({
        'success': True,
        'total_categories': len(categories),
        'categories': sorted(categories, key=lambda x: x['ingredient_count'], reverse=True)
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    rag = get_rag()
    db_stats = rag.database.get_stats()
    
    return jsonify({
        'success': True,
        'database': db_stats,
        'classifier': {
            'model': rag.classifier.model_name,
            'device': rag.classifier.device,
            'num_categories': len(rag.classifier.get_all_labels())
        },
        'ollama': {
            'available': rag.llm.is_available(),
            'model': rag.llm.model,
            'models': rag.llm.list_models() if rag.llm.is_available() else []
        }
    })

# =============================================================================
# SERVE WEB INTERFACE
# =============================================================================

@app.route('/')
def index():
    """Serve the main web interface"""
    return send_file('nutrisage_web.html')

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Get configuration from environment
    port = int(os.environ.get('PORT', 5000))
    ollama_model = os.environ.get('OLLAMA_MODEL', 'mistral')
    classifier_model = os.environ.get('CLASSIFIER_MODEL', 'facebook/bart-large-mnli')
    
    print(f"""
============================================================
           NutriSage Web Server Starting              
============================================================
  URL: http://localhost:{port}
  
  Configuration:
    Ollama Model: {ollama_model}
    Classifier: {classifier_model}
  
  API Endpoints:
    GET  /              - Web interface
    POST /api/query     - Process health query (full RAG)
    POST /api/classify  - Classify query only
    GET  /api/ingredients?category=X - Get ingredients
    GET  /api/categories - List all categories
    GET  /api/stats     - System statistics
    GET  /api/health    - Health check
============================================================
""")
    
    # Pre-initialize the RAG system
    print("Initializing RAG system (this may take a moment)...\n")
    get_rag()
    
    print("\nStarting web server...")
    app.run(host='0.0.0.0', port=port, debug=False)