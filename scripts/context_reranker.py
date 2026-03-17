"""
Context re-ranker for galago species predictions.

Uses location, season, and time-of-night to re-rank species predictions
using Bayesian priors. This follows Merlin Bird-ID's approach of using
context to improve predictions.

Usage:
    from scripts.context_reranker import rerank_predictions
    
    reranked = rerank_predictions(
        predictions=[("Paragalago_granti", 0.15), ("Galago_senegalensis", 0.12)],
        location="Tanzania",
        month=6,  # June
        hour=22   # 10 PM
    )
"""
from pathlib import Path
import json
from typing import List, Tuple, Optional

# Optional: polygon-based priors (lat/lon point in IUCN range polygons)
# Support both "run from scripts/" (plain module imports) and "import scripts.context_reranker".
IucnPolygonPrior = None  # type: ignore
try:
    # When scripts/ is on sys.path
    from iucn_polygon_priors import IucnPolygonPrior as _IucnPolygonPrior
    IucnPolygonPrior = _IucnPolygonPrior  # type: ignore
except Exception:
    try:
        # When importing as a package/module path (namespace packages)
        from scripts.iucn_polygon_priors import IucnPolygonPrior as _IucnPolygonPrior  # type: ignore
        IucnPolygonPrior = _IucnPolygonPrior  # type: ignore
    except Exception:
        IucnPolygonPrior = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANGES_FILE = PROJECT_ROOT / "data" / "species_ranges.json"

# Optional global polygon prior index (lazy-loaded)
_POLYGON_PRIOR = IucnPolygonPrior() if IucnPolygonPrior is not None else None

# Load species ranges
if RANGES_FILE.exists():
    with open(RANGES_FILE, 'r') as f:
        RANGES_DATA = json.load(f)
    SPECIES_RANGES = RANGES_DATA.get("species_ranges", {})
else:
    SPECIES_RANGES = {}
    print(f"Warning: {RANGES_FILE} not found. Location filtering disabled.")


def get_location_prior(
    species: str,
    location: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> float:
    """
    Get prior probability for species based on location.
    
    Returns:
        - If lat/lon provided and polygon priors are available:
            - 1.0 if point is inside species polygon
            - 0.1 if outside
            - 0.5 if polygons unavailable
        - Else (string-based fallback using `data/species_ranges.json`):
            - 1.0 if location matches species range (high confidence)
            - 0.7 if location is in same region (moderate confidence)
            - 0.1 if location doesn't match (low confidence)
            - 0.5 if no location provided (neutral)
    """
    # Prefer polygon prior if a point is available.
    if lat is not None and lon is not None and _POLYGON_PRIOR is not None:
        p = _POLYGON_PRIOR.prior(species, lat=lat, lon=lon)
        # If polygon data isn't available/loaded, prior() returns neutral.
        return p

    if not location or species not in SPECIES_RANGES:
        return 0.5  # Neutral prior if no location or species not in database
    
    species_data = SPECIES_RANGES[species]
    countries = species_data.get("countries", [])
    regions = species_data.get("regions", [])
    
    # Check if location matches country
    location_lower = location.lower()
    for country in countries:
        if location_lower in country.lower() or country.lower() in location_lower:
            return 1.0  # High confidence - exact match
    
    # Check if location matches region
    for region in regions:
        if location_lower in region.lower() or region.lower() in location_lower:
            return 0.7  # Moderate confidence - regional match
    
    # No match
    return 0.1


def get_seasonality_prior(species: str, month: Optional[int] = None) -> float:
    """
    Get prior probability based on seasonality.
    
    Currently returns neutral (0.5) as we don't have seasonality data.
    Can be extended with actual seasonality information if available.
    """
    # TODO: Add seasonality data if available
    # Some species may be more active in certain months
    return 0.5  # Neutral


def get_time_of_night_prior(species: str, hour: Optional[int] = None) -> float:
    """
    Get prior probability based on time of night.
    
    Galagos are nocturnal, so all species are active at night.
    Some may have peak activity times, but we don't have that data yet.
    """
    if hour is None:
        return 0.5  # Neutral
    
    # All galagos are nocturnal (active 18:00-06:00)
    if 18 <= hour or hour < 6:
        return 1.0  # High confidence during night hours
    elif 6 <= hour < 12:
        return 0.2  # Low confidence during day
    else:
        return 0.3  # Moderate during afternoon/evening transition


def rerank_predictions(
    predictions: List[Tuple[str, float]],
    location: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    month: Optional[int] = None,
    hour: Optional[int] = None,
    alpha: float = 0.5
) -> List[Tuple[str, float, dict]]:
    """
    Re-rank predictions using Bayesian priors from context.
    
    Args:
        predictions: List of (species, probability) tuples
        location: Location string (country or region)
        month: Month (1-12)
        hour: Hour of day (0-23)
        alpha: Weight for context priors (0.0 = no context, 1.0 = full context)
    
    Returns:
        List of (species, reranked_probability, metadata) tuples
        Metadata includes original prob, location_prior, and final_prior
    """
    if not predictions:
        return []
    
    reranked = []
    
    for species, original_prob in predictions:
        # Get priors
        location_prior = get_location_prior(species, location=location, lat=lat, lon=lon)
        seasonality_prior = get_seasonality_prior(species, month)
        time_prior = get_time_of_night_prior(species, hour)
        
        # Combine priors (geometric mean for multiplicative effect)
        combined_prior = (location_prior * seasonality_prior * time_prior) ** (1/3)
        
        # Apply Bayesian re-ranking: P(species|data) ∝ P(data|species) * P(species)
        # We approximate: reranked_prob = original_prob * prior^alpha
        reranked_prob = original_prob * (combined_prior ** alpha)
        
        metadata = {
            "original_prob": original_prob,
            "location_prior": location_prior,
            "seasonality_prior": seasonality_prior,
            "time_prior": time_prior,
            "combined_prior": combined_prior,
            "reranked_prob": reranked_prob,
        }
        
        reranked.append((species, reranked_prob, metadata))
    
    # Normalize probabilities to sum to 1
    total = sum(prob for _, prob, _ in reranked)
    if total > 0:
        reranked = [(species, prob / total, meta) for species, prob, meta in reranked]
    
    # Sort by reranked probability (descending)
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    return reranked


def get_location_status(species: str, location: Optional[str] = None) -> str:
    """
    Get human-readable status for location match.
    
    Returns:
        - "Most likely here" if location matches
        - "Possible but uncommon" if regional match
        - "Unlikely here" if no match
        - "Unknown" if no location provided
    """
    if not location:
        return "Unknown"

    prior = get_location_prior(species, location)
    
    if prior >= 0.9:
        return "Most likely here"
    elif prior >= 0.5:
        return "Possible but uncommon"
    else:
        return "Unlikely here"


def get_location_status_point(species: str, lat: Optional[float] = None, lon: Optional[float] = None) -> str:
    """
    Human-readable status for polygon-based location.
    Returns "Unknown" if polygons are unavailable or no point given.
    """
    if _POLYGON_PRIOR is None:
        return "Unknown"
    return _POLYGON_PRIOR.status(species, lat=lat, lon=lon)


if __name__ == "__main__":
    # Test the re-ranker
    print("Context Re-ranker Test")
    print("="*60)
    
    # Example predictions
    predictions = [
        ("Paragalago_granti", 0.15),
        ("Galago_senegalensis", 0.12),
        ("Paragalago_rondoensis", 0.10),
        ("Otolemur_crassicaudatus", 0.08),
    ]
    
    print("\nOriginal predictions:")
    for species, prob in predictions:
        print(f"  {species:30s}: {prob:.3f}")
    
    # Test with location
    print("\nWith location='Tanzania':")
    reranked = rerank_predictions(predictions, location="Tanzania", hour=22)
    for species, prob, meta in reranked:
        status = get_location_status(species, "Tanzania")
        print(f"  {species:30s}: {prob:.3f} ({status})")
        print(f"    Original: {meta['original_prob']:.3f}, Prior: {meta['combined_prior']:.3f}")
    
    # Test with different location
    print("\nWith location='Kenya':")
    reranked = rerank_predictions(predictions, location="Kenya", hour=22)
    for species, prob, meta in reranked:
        status = get_location_status(species, "Kenya")
        print(f"  {species:30s}: {prob:.3f} ({status})")
        print(f"    Original: {meta['original_prob']:.3f}, Prior: {meta['combined_prior']:.3f}")

