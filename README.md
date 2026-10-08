# Large Language Model and Knowledge Graph Collaborative Framework for Aircraft Manufacturing System Design

**Demo video:** [https://youtu.be/icpT_mcnjMk](https://youtu.be/icpT_mcnjMk)  
**Associated paper:** [https://doi.org/10.1016/j.aei.2026.105349](https://doi.org/10.1016/j.aei.2026.105349)

> **Availability notice:** The MBSE and simulation conversion components (`csv2GOPPRRE.py`, `GOPPRRE2sim.py`, and `GOPPRRE.owl`) are not included in the public repository because they contain confidential implementation details. Their absence affects only MBSE OWL generation and MATLAB SimEvents model generation. Knowledge-graph Q&A, assembly-plan generation, verification, and regeneration remain fully available. The application loads these components only when the corresponding function is requested and reports a clear error if they are unavailable.

[中文说明](README.zh-CN.md)

## Overview

This project is an intelligent aircraft fuselage assembly system design application that combines large language models (LLMs), a Neo4j knowledge graph (KG), model-based systems engineering (MBSE), and simulation-model generation. It provides a single Gradio interface for domain knowledge retrieval, automatic assembly-plan generation, engineering verification, feedback-driven regeneration, and—when the private conversion components are available—MBSE and simulation-model generation.

## Features

- Natural-language knowledge-graph Q&A with Cypher generation, execution, retry, and interactive graph visualization.
- Aircraft fuselage assembly-plan generation grounded in process, operation, resource, resource-requirement, and predecessor knowledge.
- Versioned CSV output for generated and regenerated plans.
- Deterministic verification of operation duration, resource assignment, operation cost, and peak resource usage.
- LLM-based verification against user-defined engineering constraints.
- Human-feedback-driven plan regeneration.
- Lazy loading of the confidential MBSE and simulation conversion components.
- Runtime and optional NVIDIA GPU-memory monitoring.

## Project structure

```text
.
├── src/aircraft_assembly_design/
│   ├── assets/                 # Static application assets
│   ├── ui/                     # Gradio layout, styles, and callbacks
│   ├── verification/           # Deterministic checks and LLM verification
│   ├── app.py                  # Public application entry point
│   ├── clients.py              # Lazy OpenAI-compatible and Neo4j clients
│   ├── config.py               # Environment settings and filesystem paths
│   ├── graph_visualization.py  # PyVis graph rendering
│   ├── monitoring.py           # Runtime and GPU monitoring
│   ├── plans.py                # Plan parsing, persistence, and versioning
│   ├── private_plugins.py      # Lazy adapter for non-public components
│   ├── prompts.py              # LLM prompt templates
│   ├── qa.py                   # Routing, KG Q&A, and plan generation
│   ├── regeneration.py         # Feedback-driven plan regeneration
│   └── streaming.py            # Streaming-response helpers
├── scripts/
│   └── ontology_import.cypher  # Ontology import and KG normalization script
├── pyproject.toml
└── requirements.txt
```

The application creates `plans/`, `constraints/`, `MBSE/`, `Simulation/`, `Verification/`, and `static/` as needed. These runtime directories are excluded from version control.

## Requirements

- Python 3.10–3.12
- Neo4j Community 5.26.1
- APOC 5.26.1
- Neosemantics 5.20.0
- An OpenAI-compatible API endpoint with access to `gpt-5.6-sol`, or another configured model
- NVIDIA drivers and `nvidia-smi` only for GPU-memory reporting
- MATLAB with SimEvents only for executing generated simulation models

## Installation

Create and activate a Python environment:

```powershell
conda create -n aircraft-design python=3.10 -y
conda activate aircraft-design
```

Install all dependencies and the application:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Create the local configuration file:

```powershell
Copy-Item .env.example .env
```

Configure the required connections in `.env`:

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_openai_compatible_base_url
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
AIRCRAFT_DESIGN_MODEL=gpt-5.6-sol
```

The application uses `gpt-5.6-sol` by default for higher performance. Set `AIRCRAFT_DESIGN_MODEL` to another model identifier if required by your OpenAI-compatible endpoint. The application host and port, GPU count, monitoring interval, runtime directory, and private-component directory can be configured with the other optional `AIRCRAFT_DESIGN_*` variables documented in `.env.example`. The `.env` file is ignored by version control and must not be committed.

## Neo4j setup and ontology import

The complete import workflow is available in [`scripts/ontology_import.cypher`](scripts/ontology_import.cypher). It is based on the reproduction procedure in `docs/复现流程_AEI.docx`.

### 1. Install APOC and Neosemantics

1. Download APOC 5.26.1 and Neosemantics 5.20.0.
2. Copy both JAR files to the Neo4j `plugins` directory.
3. Add or update the following entries in `neo4j.conf`:

```properties
dbms.unmanaged_extension_classes=n10s.endpoint=/rdf
dbms.security.procedures.unrestricted=apoc.*,n10s.*
dbms.security.procedures.allowlist=apoc.*,n10s.*
dbms.security.allow_csv_import_from_file_urls=true
```

4. Restart Neo4j. On Windows, Neo4j can be started with `neo4j.bat console`.
5. Verify that APOC is available:

```cypher
RETURN apoc.version();
```

### 2. Export and import the ontology

1. Export the aircraft assembly ontology from Protégé in Turtle (`.ttl`) format.
2. Open [`scripts/ontology_import.cypher`](scripts/ontology_import.cypher).
3. Replace the placeholder below with the local file URL of the exported ontology:

```text
file:///ABSOLUTE/PATH/aircraft_assembly_process_ontology.ttl
```

For example, a Windows path can be written as `file:///E:/ontology/aircraft_assembly_process_ontology.ttl`.

4. Run the script section by section in Neo4j Browser. The script:

   - initializes Neosemantics and creates the URI uniqueness constraint;
   - imports the Turtle ontology;
   - normalizes imported node names and ontology relationships;
   - assigns duration and Manual/Auto attributes to 20 operations;
   - assigns cost, calendar, and capacity attributes to 10 resources;
   - reconstructs operation-resource requirements;
   - creates the application-facing `Process`, `Operation`, and `Resource` labels;
   - converts ontology restrictions into the relationship types used by the application;
   - normalizes the final properties and resource names.

To perform a clean import, clear the database only when deletion is intended:

```cypher
MATCH (n) DETACH DELETE n;
```

To remove the Neosemantics URI constraint after import:

```cypher
DROP CONSTRAINT n10s_unique_uri;
```

### 3. Check the imported graph

Before normalization, Neosemantics may create labels including `Resource`, `_GraphConfig`, `Class`, `Property`, and `Relationship`, as well as relationship types such as `DOMAIN`, `RANGE`, `SCO`, `SCO_RESTRICTION`, and `SPO`. The import script converts these ontology-level elements into the graph schema expected by the application.

## Running the application

Start the application from the project root:

```powershell
python -m aircraft_assembly_design
```

Alternatively, use the installed command:

```powershell
aircraft-assembly-design
```

Open [http://localhost:7860](http://localhost:7860) unless a different host or port is configured.

## Usage workflow

1. Query process, operation, resource, resource-requirement, or predecessor knowledge.
2. Select **Plan** or enter a custom design request to generate an assembly plan.
3. Select **Verification** to run deterministic checks and produce an LLM verification report.
4. Enter human feedback and select **Regeneration** to produce a revised plan version.
5. If the confidential components are available in the configured private-component directory, select **MBSE** and **Simulation** to generate the corresponding OWL and MATLAB models.

## Citation

If you use this project in academic work, please cite the associated paper through its DOI: [10.1016/j.aei.2026.105349](https://doi.org/10.1016/j.aei.2026.105349).
