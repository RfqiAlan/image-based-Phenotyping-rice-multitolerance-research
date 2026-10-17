# Dataset Documentation

This directory contains the image-based phenotyping (IBP) datasets extracted from M2 mutant rice (*Oryza sativa* L.) subjected to sequential abiotic stress (Drought followed by Salinity).

## 1. Files
- `Kekeringan_Gabung.csv`: Phenotypic traits extracted at the end of the drought stress phase (14 days of progressive water deficit).
- `Salinitas_Gabung.csv`: Phenotypic traits extracted after the subsequent salinity stress phase (salinized nutrient solution up to 12 dS/m).

## 2. Plant Identity & Alignment
Both files contain **148 individual plants** (M2 mutants along with check cultivars). The rows in both datasets are matched 1-to-1 by plant identifier, enabling paired sequential stress tracking.

## 3. Trait Definitions

### A. Morphometric Traits
- `Area`: Projected shoot/canopy area ($cm^2$), representing total vegetative biomass.
- `Perim`: Perimeter of the canopy boundary ($cm$), quantifying boundary complexity and leaf edge expansion.
- `Widht`: Maximum lateral canopy diameter / canopy width ($cm$).
- `Height`: Maximum vertical canopy height ($cm$).
- `Major`: Length of the primary ellipse axis fitted to the canopy ($cm$).
- `Minor`: Length of the secondary ellipse axis fitted to the canopy ($cm$).
- `Circle`: Circularity score ($4\pi \times \frac{\text{Area}}{\text{Perimeter}^2}$), reflecting geometric compactness and drought-induced leaf rolling / wilted morphology.
- `Solidity`: Canopy solidity ($\frac{\text{Area}}{\text{Convex Area}}$), measuring internal canopy density and spatial leaf arrangement.

### B. Colorimetric Traits (Vegetation Indices)
- `GLI`: Green Leaf Index ($\frac{2G - R - B}{2G + R + B}$), sensitive to chlorophyll retention and senescence.
- `NGRDI`: Normalized Green-Red Difference Index ($\frac{G - R}{G + R}$), tracking chlorosis and anthocyanin accumulation.
- *Note:* VARI was excluded during multicollinearity screening due to near-perfect collinearity with NGRDI ($r = 0.99$, VIF $> 10$).

### C. Ground-Truth Stress Scoring
- `Luma_Y`: Standard Evaluation System (SES) score assigned according to IRRI guidelines (1 = Highly Tolerant, 3 = Tolerant, 5 = Moderately Tolerant, 7 = Susceptible, 9 = Highly Susceptible/Dead).
- **Multiple Tolerance Score (MTS)**:
  - **Class 1 (Dual-Tolerant)**: $\text{SES}_{\text{Drought}} \le 3$ and $\text{SES}_{\text{Salin}} \le 3$ (Elite candidates).
  - **Class 3 (Mono-Tolerant)**: Tolerant to only one stress ($\text{SES} \le 3$ in one stage and $> 3$ in the other).
  - **Class 5 (Dual-Susceptible)**: Susceptible to both stresses ($\text{SES}_{\text{Drought}} > 3$ and $\text{SES}_{\text{Salin}} > 3$).
