# FEVER Relation Attribution Results

FEVER-2K stratified 70/30 train/test split. This setting provides reliable Supported / Insufficient / Contradicted relation labels and is used to validate the attribution side of SCAD-RAG.

| Method | Accuracy | Relation Macro-F1 | Supported-F1 | Insufficient-F1 | Contradicted-F1 |
|---|---:|---:|---:|---:|---:|
| SCAD-RAG-Rule | 0.5517 | 0.5528 | 0.6352 | 0.3899 | 0.6333 |
| NLI-rule | 0.6017 | 0.5774 | 0.7088 | 0.3899 | 0.6333 |
| LogReg over SCAD features | 0.6833 | 0.6433 | 0.8007 | 0.4789 | 0.6505 |
| HistGB over SCAD features | 0.7183 | 0.6727 | 0.8219 | 0.5241 | 0.6719 |
| RF over SCAD features | **0.7317** | **0.6944** | **0.8288** | **0.5361** | **0.7183** |

Interpretation: SCAD diagnostic features provide stronger supervised relation attribution than NLI-rule and rule-only variants when reliable relation labels are available.
