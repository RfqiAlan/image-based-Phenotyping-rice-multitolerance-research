# Sequential Drought and Salinity Tolerance Screening in Rice using Image-Based Phenotyping and Machine Learning

This repository provides the core source code and phenotypic datasets accompanying the manuscript:  
> **"Image-Based Phenotyping and Machine Learning for High-Throughput Screening of Sequential Drought and Salinity Tolerance in M2 Mutant Rice (*Oryza sativa* L.)"**  
> *(Prepared for submission)*

---

## 📂 Repository Structure

```text
├── data/
│   ├── Kekeringan_Gabung.csv              # Phenotypic data under drought stress phase
│   ├── Salinitas_Gabung.csv               # Phenotypic data under salinity stress phase
│   └── README.md                          # Data dictionary and trait descriptions
├── notebooks/
│   └── IBP_Multitolerance_Workflow.ipynb  # Interactive analysis workflow
├── src/
│   ├── data_loader.py                     # Data ingestion, preprocessing, and MTS construct
│   ├── feature_selection.py               # Statistical screening and factor analysis
│   ├── models.py                          # Model configurations (RF, GBM, SVM, Ordinal)
│   └── evaluation.py                      # Nested cross-validation and visualization functions
├── main.py                                # End-to-end execution script
├── requirements.txt                       # Required Python libraries
└── LICENSE                                # MIT License
```

---

## ⚙️ Installation

Python 3.9 or newer is recommended.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RfqiAlan/image-based-Phenotyping-rice-multitolerance-research.git
   cd image-based-Phenotyping-rice-multitolerance-research
   ```

2. **(Optional) Set up a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Run the Full Pipeline
Execute the main script to reproduce the repeated stratified nested cross-validation and feature ablation experiments:
```bash
python main.py
```
*Note: Generated performance tables (CSV) and figures (PNG) will be saved locally to the `results/` and `figures/` directories.*

### 2. Interactive Notebook
For step-by-step exploration of the workflow and data visualization:
```bash
jupyter notebook notebooks/IBP_Multitolerance_Workflow.ipynb
```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
