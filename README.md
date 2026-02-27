# AnchorMap

AnchorMap is a schema-driven multi-agent pipeline for standardising heterogeneous engineering variable labels to canonical identifiers.

It targets cases where the same physical quantity is named differently across sources such as OEM datasheets, simulation/co-simulation models, and technical or regulatory documentation. AnchorMap combines embedding-based candidate retrieval with conditional large language model (LLM) reasoning and confidence-based routing:

- **Accept** — automatically map to a canonical identifier  
- **Human Review** — defer ambiguous cases  
- **No Match** — abstain when no suitable schema term exists  

The canonical vocabulary is defined in a JSON-LD schema aligned with the Vessel Information Structure (VIS), enabling consistent variable-level interoperability for maritime co-simulation and collaborative design workflows.

Repository:  
https://github.com/saadahmedrana/AnchorMap

---





## 🎬 Demo Video

[![AnchorMap Demo](demo.gif)](https://github.com/user-attachments/assets/abd39406-ec94-489b-a83b-7ac3fff74eaa)

Direct link: https://github.com/user-attachments/assets/abd39406-ec94-489b-a83b-7ac3fff74eaa


# 🚀 Quick Start

Follow these steps to get AnchorMap running locally.

---

## 1️⃣ Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/saadahmedrana/AnchorMap.git
cd AnchorMap
```

---

## 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure API Key

Create a `.env` file in the project root directory:

```bash
touch .env        # macOS / Linux
```

Add the following line inside the file:

```bash
AALTO_KEY="YOUR_API_KEY_HERE"
```

This key is required for:

- Embedding generation
- LLM-based reasoning
- Candidate selection

---

# ▶ Running AnchorMap

After installation and configuration, you have **two options**:

---

## Option 1 — Run the Pipeline (Terminal)

```bash
python run_pipeline.py
```

This executes the full AnchorMap pipeline:

- Schema indexing  
- Embedding-based candidate retrieval  
- Conditional LLM reasoning  
- Confidence-based routing  
- Variable renaming in Turtle (.ttl) files  


---

## Option 2 — Run the GUI (Streamlit Web App)

Launch the interactive web interface with:

```bash
streamlit run run_app.py
```

The app will start locally, typically at:

```
http://localhost:8501
```

Inside the GUI you can:

- Upload OEM Turtle (.ttl) files
- Inspect retrieved candidates and similarity scores
- View routing decisions (Accept / Human Review / No Match)
- Manually approve ambiguous mappings
- Download updated schema-aligned outputs

This mode is recommended for:
- Interactive inspection
- Human-in-the-loop review
- Demonstrations

---

# 🧠 Pipeline Architecture Overview

AnchorMap consists of four core agents:

### 1. Schema Indexing Agent  
Builds the embedding-based anchor space from the JSON-LD schema.

### 2. Retrieval & Reasoning Agent  
Retrieves Top-K candidates using cosine similarity and conditionally invokes an LLM when ambiguity is detected.

### 3. Confidence Routing Agent  
Applies dual thresholds to route each mapping to:

- Accept  
- Human Review  
- No Match  

### 4. Variable Renaming Agent  
Rewrites accepted variable labels in the OEM Turtle (.ttl) file while preserving RDF structure and metadata.

---

# 📂 Repository Structure (Core Components)

- `JSONSCHEMAFORSHIP.jsonld` — Canonical JSON-LD schema (Single Source of Truth)
- `embed_SCHEMA.py` — Schema indexing and embedding generation
- `iterative_masteragent.py` — used for sweep of human review threshold parameter
- `masteragent_ecms.py` — Pipeline orchestration
- `variable_renamer_agent.py` — Rewrites accepted labels in TTL files
- `eval_results_ecms.py` — Evaluation scripts
- `human_review_cli.py` — Command-line human review interface
- `clean_csv.py` — Preprocessing helper
- `skip_variables.csv` — Variables excluded from evaluation
- `CorrectNamesMappings.xlsx` — Mapping reference file for evaluating accuracy of pipeline

---

# 📝 Notes

- The JSON-LD schema is treated as the authoritative vocabulary.
- Vendor-specific or non-standard variables are intentionally routed to **No Match**.
- Human review is a deliberate safety mechanism for deployment in simulation and compliance workflows.
- The pipeline is tested with **Python 3.11** (compatible with 3.10+).

---

# 📖 Citation

If you use AnchorMap in academic work, please cite the associated ECMS publication (details available in the repository).

---

# 📜 License

See repository for license information.