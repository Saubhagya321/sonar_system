"""
Explain where pairs are filtered: 10,000 → 8,401
This happens in the DATA PREPARATION phase, NOT in the API filtering
"""

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    WHERE PAIRS ARE FILTERED OUT                              ║
║                    (10,000 → 8,401 reduction)                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

The reduction from 10,000 to 8,401 pairs happens BEFORE the API even starts.
This is in the DATA PREPARATION / FEATURE ENGINEERING phase.

═══════════════════════════════════════════════════════════════════════════════
FILTERING PIPELINE:
═══════════════════════════════════════════════════════════════════════════════

PHASE 1: DATA PREPARATION (preprocessing/feature_engineering.py)
────────────────────────────────────────────────────────────────────────────────

1. create_pair_features() in feature_engineering.py
   ├─ Reads raw booking data
   ├─ Creates ChauffeurVehiclePair IDs (format: "chauffeur_id_vehicle_id")
   ├─ FILTERS OUT pairs with invalid IDs:
   │  ├─ Invalid chauffeur IDs (NaN, non-numeric)
   │  ├─ Invalid vehicle IDs (NaN, non-numeric)
   │  └─ Invalid pair format (not matching "123_456" pattern)
   └─ Result: ~10,000 pairs with valid IDs

   FILE: preprocessing/feature_engineering.py
   LINE: ~50-150 (create_pair_features method)

2. create_interaction_matrix() in feature_engineering.py  
   ├─ Takes pairs from step 1
   ├─ FILTERS by minimum interactions:
   │  ├─ User must have >= MIN_INTERACTIONS (default: 2)
   │  └─ Item must have >= MIN_INTERACTIONS (default: 2)
   └─ Result: ~8,401 pairs that meet interaction threshold

   FILE: preprocessing/feature_engineering.py
   LINE: ~400-450 (create_interaction_matrix method)
   CONFIG: config/config.py → MIN_INTERACTIONS = 2

PHASE 2: API FILTERING (api/filters.py)
────────────────────────────────────────────────────────────────────────────────

3. filter_candidates() in filters.py
   ├─ Takes 8,401 pairs from trained model
   ├─ FILTERS by request requirements:
   │  ├─ Hard requirements (vehicle type, capacity, level)
   │  ├─ Preferences (excluded chauffeurs/vehicles)
   │  └─ Quality filters (rating, recency)
   └─ Result: ~50 pairs matching your specific request

   FILE: api/filters.py
   LINE: ~180-250 (filter_candidates method)

═══════════════════════════════════════════════════════════════════════════════
WHY THE REDUCTION HAPPENS:
═══════════════════════════════════════════════════════════════════════════════

REASON 1: INVALID PAIR IDs (10,000 → ~9,900)
────────────────────────────────────────────────────────────────────────────────
Location: preprocessing/feature_engineering.py → create_pair_features()

Example problematic pairs removed:
├─ "NaN_1269" - chauffeur ID is NaN
├─ "2681_NaN" - vehicle ID is NaN  
├─ "abc_def" - non-numeric IDs
└─ "123" - wrong format (needs "123_456")

CODE that does this:
────────────────────────────────────────────────────────────────────────────────
# In feature_engineering.py, line ~90

# Convert to numeric, coercing errors to NaN
df['RefChauffeur'] = pd.to_numeric(df['RefChauffeur'], errors='coerce')
df['RefVehicle'] = pd.to_numeric(df['RefVehicle'], errors='coerce')

# Drop rows with invalid IDs
df = df.dropna(subset=['RefChauffeur', 'RefVehicle'])

# Ensure IDs are integers
df['RefChauffeur'] = df['RefChauffeur'].astype(int)
df['RefVehicle'] = df['RefVehicle'].astype(int)

# Create pair identifier
df['ChauffeurVehiclePair'] = (
    df['RefChauffeur'].astype(str) + '_' + 
    df['RefVehicle'].astype(str)
)
────────────────────────────────────────────────────────────────────────────────

REASON 2: MINIMUM INTERACTION FILTER (~9,900 → 8,401)
────────────────────────────────────────────────────────────────────────────────
Location: preprocessing/feature_engineering.py → create_interaction_matrix()

Removes pairs with too few interactions (< 2 bookings by default)

WHY? To ensure quality recommendations:
├─ Pairs with 1 booking are unreliable
├─ Users with 1 booking can't show preferences
└─ Items with 1 rating can't be properly scored

CODE that does this:
────────────────────────────────────────────────────────────────────────────────
# In feature_engineering.py, line ~430

min_interactions = min_interactions or self.config.MIN_INTERACTIONS  # = 2

# Filter by minimum interactions
user_counts = ratings_df['RefClient'].value_counts()
item_counts = ratings_df['ChauffeurVehiclePair'].value_counts()

valid_users = user_counts[user_counts >= min_interactions].index
valid_items = item_counts[item_counts >= min_interactions].index

filtered_ratings = ratings_df[
    (ratings_df['RefClient'].isin(valid_users)) &
    (ratings_df['ChauffeurVehiclePair'].isin(valid_items))
].copy()
────────────────────────────────────────────────────────────────────────────────

═══════════════════════════════════════════════════════════════════════════════
CHECKING VEHICLE 1269:
═══════════════════════════════════════════════════════════════════════════════

To find out if vehicle 1269 is in the 8,401 pairs:

Option 1: Run the trace script
────────────────────────────────────────────────────────────────────────────────
python trace_filtering.py

This will tell you:
   If vehicle 1269 is in the final 8,401 pairs
   If vehicle 1269 has NULL/invalid vehicle type
   How many pairs use vehicle 1269

Option 2: Check manually
────────────────────────────────────────────────────────────────────────────────
import pandas as pd

# Load final pairs
matrix = pd.read_pickle('output/interaction_matrix.pkl')
pairs_with_1269 = [p for p in matrix.columns if p.endswith('_1269')]

print(f"Pairs with vehicle 1269: {len(pairs_with_1269)}")
print(pairs_with_1269)

If this returns 0: Vehicle 1269 has < 2 bookings (filtered out)
If this returns >0: Vehicle 1269 IS in the 8,401 pairs

═══════════════════════════════════════════════════════════════════════════════
NULL VEHICLE TYPES IN API:
═══════════════════════════════════════════════════════════════════════════════

The "vehicle_type_not_indexed: 34" in your logs means:

Location: api/filters.py → _apply_hard_requirements()

34 pairs where:
├─ Vehicle ID is valid
├─ RefVehicleType is a number (e.g., 51)
└─ BUT that vehicle type ID doesn't exist in vehicle_type_final.csv

These are CORRECTLY rejected in strict mode.

To find which vehicle IDs have this issue:
────────────────────────────────────────────────────────────────────────────────
python trace_filtering.py

This will output:
   Vehicle IDs with invalid RefVehicleType (type doesn't exist):
   [123, 456, 789, ...]  ← These are the problematic vehicle IDs
   
It will also save them to: problematic_vehicles.json

═══════════════════════════════════════════════════════════════════════════════
SOLUTION TO GET VEHICLE 1269 IN RESULTS:
═══════════════════════════════════════════════════════════════════════════════

PROBLEM: Vehicle 1269 (type 51) not appearing in results for type 51 request

Possible reasons:

1. Vehicle 1269 has < 2 bookings
   └─ SOLUTION: Lower MIN_INTERACTIONS in config.py (or add more bookings)

2. Vehicle 1269 doesn't exist in vehicle_final.csv  
   └─ SOLUTION: Add vehicle 1269 to the CSV

3. Vehicle 1269 exists but has NULL RefVehicleType
   └─ SOLUTION: Set RefVehicleType = 51 for vehicle 1269

4. Vehicle 1269 exists but was filtered in data preparation
   └─ SOLUTION: Check booking data - does 1269 have valid bookings?

RUN THIS TO DIAGNOSE:
────────────────────────────────────────────────────────────────────────────────
python trace_filtering.py

This will show EXACTLY where vehicle 1269 is being filtered out.

═══════════════════════════════════════════════════════════════════════════════
""")

print("\n  Run this script now to diagnose:")
print("   python trace_filtering.py")
print("\n It will show you:")
print("      Whether vehicle 1269 is in the 8,401 pairs")
print("      All vehicles with NULL vehicle types") 
print("      All vehicles with invalid vehicle types")
print("      Which pairs are being filtered and why")
print("\n" + "="*80)
