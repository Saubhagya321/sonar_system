# Recommendation Combiner System

A production-ready system for combining multiple recommendation models using RRF (Reciprocal Rank Fusion) and weighted score fusion.

##    Overview

This system combines 4 recommendation models:
1. **LightGCN** (Heterogeneous Graph Neural Network)
2. **Collaborative Filtering** 
3. **Item-Item Similarity**
4. **User-Item Context Similarity**

### Two Main Pipelines

1. **Grid Search Pipeline** (Monthly/Training)
   - Optimizes weights for each client type
   - Uses NDCG@10 for validation
   - Saves optimal weights per client type

2. **Request Pipeline** (Real-time)
   - Generates recommendations for incoming requests
   - Uses pre-optimized weights
   - Handles cold-start scenarios
   - Applies business filters and context boosting

## 📁 Project Structure

```
recommendation_combiner/
├── config.py                 # Configuration and settings
├── data_loaders.py          # Data loading functions (one per model)
├── fusion_models.py         # RRF and score fusion logic
├── grid_search.py           # Monthly weight optimization
├── request_pipeline.py      # Real-time recommendation generation
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── data/                   # Data directory
│   ├── grid_search/       # CSV files for training
│   │   ├── lightgcn_recommendations.csv
│   │   └── collaborative_recommendations.csv
│   │
│   └── requests/          # Request data
│       ├── lightgcn_request_{client_id}.json
│       ├── collaborative_request_{client_id}.json
│       ├── item_item_similarity_{client_id}.csv
│       └── user_item_similarity_{client_id}.csv
│
├── weights/               # Saved optimal weights
│   ├── weights_CORPORATE.json
│   ├── weights_GLOBAL_CASH.json
│   ├── weights_GLOBAL_CREDIT.json
│   └── weights_UNKNOWN.json
│
└── logs/                 # Log files
```

##    Setup

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn tqdm
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

#### For Grid Search (Monthly Training):

Place your CSV files in `data/grid_search/`:

**lightgcn_recommendations.csv**
```csv
client_node_id,vehicle_idx,chauffeur_idx,rank,combined_score,cv_score,cc_score,vc_score,original_client_id,recommendation_type
0.0,34.0,42.0,1,5.369162559509277,4.604146957397461,5.203986644744873,5.53433895111084,4,lightgcn
```

**collaborative_recommendations.csv**
```csv
client_id,chauffeur_vehicle_pair,predicted_rating,confidence,rank
4,2923_1453,3.45,0.498,1
```

#### For Real-time Requests:

Place your files in `data/requests/`:

**lightgcn_request_811.json**
```json
{
  "client_id": 811,
  "pairs": [
    {
      "rank": 1,
      "vehicle_id": 122,
      "chauffeur_id": 100,
      "cv_score": 4.73,
      "cc_score": 5.17,
      "vc_score": 4.65
    }
  ]
}
```

**item_item_similarity_811.csv**
```csv
RefChauffeur,RefVehicle,similarity
4.0,356.0,0.9999999999999998
```

**user_item_similarity_811.csv**
```csv
RefVehicle,RefChauffeur,similarity
620.0,1436.0,0.7348276404219967
```

### 3. Create Directories

The system will auto-create directories, but you can also create them manually:

```bash
mkdir -p data/grid_search data/requests weights logs
```

## 📊 Usage

### Option 1: Quick Test

```bash
python main.py
```

This will:
1. Test all data loaders
2. Test fusion logic
3. Run grid search (if data available)
4. Generate example recommendations

### Option 2: Individual Components

#### Test Data Loaders

```python
from data_loaders import DataLoader

# Test grid search loaders
data = DataLoader.load_all_gridsearch_data()

# Test request loaders
request_data = DataLoader.load_all_request_data(client_id=811)
```

#### Test Fusion

```python
from fusion_models import RecommendationFusion

fusion = RecommendationFusion()

# Load your data
# ... 

# Fuse recommendations
fused = fusion.fuse_recommendations(recommendations, weights)
```

#### Run Grid Search (Monthly Optimization)

```python
from grid_search import run_monthly_optimization

# This will optimize weights and save them
run_monthly_optimization()
```

Or from command line:
```bash
python grid_search.py
```

#### Generate Real-time Recommendations

```python
from request_pipeline import RecommendationPipeline

pipeline = RecommendationPipeline()

# Simple request
result = pipeline.generate_recommendations(client_id=811)

# Request with context
context = {
    'important_trip': True,
    'red_zone_trip': True,
    'vip_client': True,
    'required_vehicle_type': 9
}
result = pipeline.generate_recommendations(
    client_id=811,
    request_context=context
)

# Get top-k
top_5 = pipeline.get_top_k(result, k=5)
```

Or from command line:
```bash
python request_pipeline.py
```

## 🔧 Configuration

Edit `config.py` to customize:

### RRF Settings
```python
RRF_K = 60  # RRF constant
```

### Grid Search Settings
```python
GRID_SEARCH_CONFIG = {
    'alpha_ranges': {
        'alpha1_rrf': [0.1, 0.2, 0.3, 0.4, 0.5],
        'alpha2_gcn': [0.1, 0.2, 0.3, 0.4, 0.5],
        'alpha3_item_sim': [0.05, 0.1, 0.15, 0.2],
        'alpha4_user_sim': [0.05, 0.1, 0.15, 0.2],
        'alpha5_confidence': [0.05, 0.1, 0.15, 0.2]
    },
    'validation_k': 10,  # NDCG@10
    'min_interactions': 5
}
```

### Business Rules
```python
BUSINESS_RULES = {
    'max_recommendations': 20,
    'min_confidence_threshold': 0.3,
    'min_combined_score': 3.0
}
```

## 📈 How It Works

### Fusion Formula

```
FinalScore = α1 × RRF 
           + α2 × (C-C + C-V + V-C) / 3
           + α3 × item_similarity
           + α4 × user_similarity
           + α5 × cf_confidence
```

Where:
- **RRF**: Reciprocal Rank Fusion = Σ(1 / (60 + rank_i))
- **C-C, C-V, V-C**: LightGCN scores (client-client, client-vehicle, vehicle-client)
- **item_similarity**: Item-item similarity score
- **user_similarity**: User-item context similarity score
- **cf_confidence**: Collaborative filtering confidence

### Cold-Start Handling

For clients with no history:
1. Uses item-item and user-item similarity
2. Computes cold-start score = 0.5 × item_sim + 0.5 × user_sim
3. Falls back to popular pairs if no similarity data

### Client Type Segmentation

Weights are optimized separately for:
- **CORPORATE** clients
- **GLOBAL_CASH** clients  
- **GLOBAL_CREDIT** clients
- **UNKNOWN** (average of the three)

## 🔄 Monthly Workflow

### 1. Collect Data
```bash
# Export recommendations from all models to CSV
# Place in data/grid_search/
```

### 2. Run Optimization
```bash
python grid_search.py
```

This will:
- Load historical recommendations
- Split by client type
- Grid search optimal weights
- Save weights to `weights/` directory

### 3. Deploy Weights
```bash
# Weights are now in weights/ and ready to use
# The request pipeline will automatically load them
```

##    Real-time Request Flow

```
1. Request arrives → client_id + context
                    ↓
2. Load client type → CORPORATE/CASH/CREDIT
                    ↓
3. Load weights → optimal α1..α5 for that type
                    ↓
4. Load 4 models → LightGCN + CF + Similarities
                    ↓
5. Compute RRF → from ranks
                    ↓
6. Fuse scores → weighted combination
                    ↓
7. Context boost → based on request context
                    ↓
8. Apply filters → business rules
                    ↓
9. Return top-k → final recommendations
```

## 📝 Example Output

```json
{
  "client_id": 811,
  "client_type": "CORPORATE",
  "is_cold_start": false,
  "total_recommendations": 15,
  "processing_time_seconds": 0.124,
  "recommendations": [
    {
      "rank": 1,
      "pair_id": "100_122",
      "final_score": 5.234,
      "rrf_score": 0.032,
      "gcn_avg_score": 4.85,
      "item_similarity": 0.92,
      "user_similarity": 0.88,
      "cf_confidence": 0.75,
      "vehicle_id": 122,
      "chauffeur_id": 100,
      "vehicle_make": "MERCEDES",
      "vehicle_model": "S CLASS"
    }
  ]
}
```

## 🐛 Troubleshooting

### File Not Found Errors
```
Check that your data files are in the correct directories:
- data/grid_search/ for training data
- data/requests/ for request data
```

### No Recommendations Generated
```
Check that:
1. Data files have the correct columns
2. Client ID exists in your data
3. pair_id columns are formatted correctly (chauffeur_vehicle)
```

### Weight Optimization Takes Too Long
```
Reduce grid search ranges in config.py:
- Use fewer alpha values
- Use smaller validation set
```

## 🔐 Production Considerations

### 1. Database Integration
Replace mock client type lookup in `request_pipeline.py`:
```python
def get_client_type(self, client_id: int) -> str:
    # TODO: Replace with actual DB query
    query = "SELECT client_type FROM clients WHERE id = ?"
    result = execute_query(query, client_id)
    return result['client_type']
```

### 2. Caching
Add Redis caching for:
- Loaded weights
- Recent recommendations
- Client type lookups

### 3. Monitoring
Add logging and metrics:
- Request latency
- Recommendation quality (CTR)
- Cold-start rate
- Weight performance

### 4. API Integration
Wrap in Flask/FastAPI:
```python
@app.post("/recommendations")
def get_recommendations(request: RecommendationRequest):
    pipeline = RecommendationPipeline()
    result = pipeline.generate_recommendations(
        client_id=request.client_id,
        request_context=request.context
    )
    return result
```

## 📞 Support

For issues or questions, check:
1. Data file formats match examples
2. All dependencies installed
3. Directories exist and have correct permissions

## 🎉 Quick Start Checklist

- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Create directories (`mkdir -p data/grid_search data/requests weights logs`)
- [ ] Place grid search CSV files in `data/grid_search/`
- [ ] Place request files in `data/requests/`
- [ ] Run test: `python main.py`
- [ ] Run grid search: `python grid_search.py`
- [ ] Generate recommendations: `python request_pipeline.py`
- [ ] Integrate with your system

## 📄 License

MIT License - Feel free to use and modify for your needs.
