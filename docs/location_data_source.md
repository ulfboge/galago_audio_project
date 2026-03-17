# Location Data Source Documentation

## Overview

The location data for species re-ranking is stored in `data/species_ranges.json`. This file contains geographic range information for each galago species, which is used by the context re-ranker to improve predictions based on recording location.

## Data Source

**Important**: The location data in `species_ranges.json` was **manually curated** based on:
- IUCN Red List range descriptions
- Published field guides and species accounts
- General knowledge of galago distributions

**This is NOT automatically downloaded or updated** - it's a static file that should be reviewed and updated manually as needed.

## File Structure

The JSON file contains:
- `countries`: List of countries where the species occurs
- `regions`: Broad geographic regions (e.g., "East Africa", "West-Central Africa")
- `notes`: Additional information about the species

## How It's Used

The context re-ranker (`scripts/context_reranker.py`) uses this data to:
1. Check if a recording location matches known species ranges
2. Adjust prediction probabilities based on geographic likelihood
3. Flag predictions as "Most likely here" or "Unlikely here"

## Updating the Data

To improve location-based re-ranking:

1. **Review accuracy**: Check if the countries/regions listed are correct for each species
2. **Add missing data**: Some species may have incomplete range information
3. **Add seasonality**: The structure supports `active_months` and `time_of_night` fields (not yet populated)
4. **Add precision**: Consider adding more specific regions or habitat types

## Example Entry

```json
{
  "Paragalago_rondoensis": {
    "countries": ["Tanzania"],
    "regions": ["East Africa"],
    "notes": "Rondo dwarf galago - restricted range"
  }
}
```

## Current Status

- ✅ Location matching implemented
- ⏳ Seasonality data not yet populated
- ⏳ Time-of-night preferences not yet populated

## Recommendations

1. **Verify all entries** against authoritative sources (IUCN Red List, field guides)
2. **Add seasonality data** if available (breeding seasons, activity patterns)
3. **Add time-of-night preferences** if known (early night vs. late night activity)
4. **Consider habitat types** for more precise matching (forest types, elevation ranges)

---

**Last Updated**: December 18, 2025  
**Status**: Manual curation, needs review and expansion
