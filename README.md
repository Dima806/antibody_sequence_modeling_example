# antibody_sequence_modeling_example
Example project for predicting biophysical properties of antibody CDR (Complementarity-Determining Region) sequences

## Downloading the OAS Data

Full paired OAS data is sourced from HuggingFace:
[bloyal/oas-paired-sequence-data](https://huggingface.co/datasets/bloyal/oas-paired-sequence-data)

```bash
# Install the HuggingFace datasets library (included in pyproject.toml)
pip install datasets>=2.19

# Download and preprocess the training split
python data/download.py --output data/full/

# Download a specific split (train / validation / test)
python data/download.py --output data/full/ --split validation
```

The script will:
1. Stream the dataset from HuggingFace via `datasets.load_dataset()`
2. Extract CDR-H3 sequences and heavy chain sequences
3. Compute Kyte-Doolittle hydrophobicity (GRAVY) and length class labels
4. Save the preprocessed data to `data/full/sequences_full.csv`

For smoke testing on CPU (Codespaces), the committed smoke dataset at
`data/smoke/sequences_smoke.csv` (~2K synthetic sequences) is used instead.
