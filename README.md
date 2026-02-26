# AnchorMap

AnchorMap is a schema-driven pipeline for standardising heterogeneous engineering variable labels to canonical identifiers.

It targets cases where the same physical quantity is named differently across sources such as OEM datasheets, simulation/co-simulation models, and technical or regulatory documentation. AnchorMap combines candidate retrieval from a VIS-aligned JSON-LD schema with conditional reasoning and confidence-based routing:

- **Accept**: auto-map to a canonical identifier  
- **Human Review**: defer ambiguous cases  
- **No Match**: abstain when no suitable schema term exists  

## Repository contents (current)
- `JSONSCHEMAFORSHIP.jsonld` — canonical schema (JSON-LD)
- `embed_SCHEMA.py` — schema embedding / indexing utilities
- `iterative_masteragent.py` — main pipeline logic
- `eval_results_ecms.py` — evaluation / analysis scripts
- `clean_csv.py`, `skip_variables.csv` — preprocessing helpers
- `Input Files/` — example inputs (TTL)
- `CorrectNamesMappings.xlsx` — mapping reference file

## Quick start
```bash
pip install -r requirements.txt
# then run the pipeline / evaluation scripts as needed
