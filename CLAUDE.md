# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an educational MCMC (Markov Chain Monte Carlo) tutorial repository written in Japanese. It consists of 6 comprehensive Jupyter notebooks that systematically teach MCMC theory and implementation from basics to advanced techniques.

## Project Structure

- **Jupyter Notebooks**: 6 sequential chapters covering MCMC fundamentals through advanced methods
  - `chapter1_mcmc_basics.ipynb`: MCMC foundations and Markov chain theory
  - `chapter2_metropolis_hastings.ipynb`: Metropolis-Hastings algorithm implementation
  - `chapter3_gibbs_sampling.ipynb`: Gibbs sampling for conditional distributions
  - `chapter4_convergence_diagnostics.ipynb`: Convergence diagnostics and performance evaluation
  - `chapter5_practical_applications.ipynb`: Bayesian regression and hierarchical models
  - `chapter6_advanced_mcmc.ipynb`: HMC, NUTS, and practical libraries
- **Documentation**: `docs/` contains supplementary markdown content
- **Dependencies**: Managed via `pyproject.toml` with scientific Python stack

## Environment Setup

### Primary method (using uv - recommended):
```bash
uv sync
uv run jupyter notebook
```

### Alternative method (using pip):
```bash
python -m venv mcmc_env
source mcmc_env/bin/activate  # Windows: mcmc_env\Scripts\activate
pip install numpy matplotlib seaborn scipy pandas scikit-learn statsmodels jupyter ipykernel plotly tqdm
jupyter notebook
```

## Core Dependencies

The project uses the standard scientific Python ecosystem:
- **numpy**: Numerical computing foundation
- **matplotlib/seaborn**: Statistical visualization
- **scipy**: Scientific computing (distributions, optimization)
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning utilities
- **statsmodels**: Statistical modeling
- **jupyter**: Notebook environment
- **plotly**: Interactive visualization
- **tqdm**: Progress bars

## Code Patterns and Conventions

### Notebook Structure
- Each notebook follows consistent pedagogical structure: theory → implementation → examples → exercises
- Code cells use standard scientific Python imports pattern
- Japanese language used for markdown explanations with English variable names
- Visualizations use consistent matplotlib/seaborn styling with `plt.rcParams['font.family'] = 'DejaVu Sans'`

### Common Function Patterns
- MCMC implementations follow functional programming style with clear parameter documentation
- Probability distributions handled via `scipy.stats` or custom implementations
- Visualization functions consistently return matplotlib figure objects
- Random seed setting (`np.random.seed(42)`) for reproducibility

### Educational Code Style
- Functions include detailed docstrings explaining mathematical concepts
- Exercise sections left partially implemented for learner completion
- Clear separation between theoretical examples and practical applications

## Key Architectural Components

### MCMC Algorithm Implementation Pattern
Each MCMC algorithm follows a consistent structure:
1. **Parameter initialization**: Setting up initial values and hyperparameters
2. **Iteration loop**: Main sampling loop with convergence monitoring
3. **Diagnostic computation**: Built-in convergence and efficiency metrics
4. **Visualization helpers**: Standardized plotting functions for trace plots, autocorrelation, etc.

### Chapter Dependencies and Learning Flow
- **Chapter 1**: Foundational concepts (Markov chains, detailed balance)
- **Chapter 2**: Core MH algorithm - builds on Chapter 1 theory
- **Chapter 3**: Gibbs sampling - uses conditional distributions, builds on MH concepts
- **Chapter 4**: Diagnostics - applies to all previous algorithms
- **Chapter 5**: Applications - integrates all previous methods with real data
- **Chapter 6**: Advanced methods - extends core concepts

### Mathematical Notation Standards
- Greek letters (θ, π, α, β) for parameters
- Bold notation (**x**, **β**) for vectors/matrices in LaTeX
- Consistent use of subscripts: x^(t) for time, x_i for indexing
- Log-scale computations for numerical stability

## MCMC-Specific Implementation Patterns

### Sampling Functions
All MCMC samplers return: `(samples, acceptance_rate, diagnostics_dict)`
- `samples`: numpy array of shape (n_iterations, n_params)
- `acceptance_rate`: float between 0 and 1
- `diagnostics_dict`: contains convergence metrics, effective sample size, etc.

### Convergence Diagnostics
Standard diagnostic workflow includes:
- R-hat (Gelman-Rubin) statistics for multi-chain convergence
- Effective sample size calculations
- Autocorrelation function analysis
- Trace plot visualization
- Accept/reject ratio monitoring

### Visualization Standards
Diagnostic plots follow consistent layout:
- Trace plots: time series of parameter evolution
- Posterior distributions: histograms with true values overlaid
- Autocorrelation plots: lag vs correlation with 5% significance lines
- Multi-panel figures for comprehensive analysis

## Development Workflow

Since this is an educational repository focused on Jupyter notebooks:
- No formal testing framework (typical for educational content)
- No linting configuration (notebooks are self-contained)
- Changes should preserve educational flow and mathematical accuracy
- When modifying notebooks, ensure mathematical formulas in LaTeX remain correctly formatted
- Maintain consistent Japanese/English language usage patterns

## Working with Notebooks

Use `NotebookRead` and `NotebookEdit` tools for notebook modifications. The notebooks contain:
- Mathematical theory in markdown cells with LaTeX equations
- Implementation examples with detailed comments
- Interactive visualizations
- Exercise problems for learners

When editing, preserve the pedagogical structure and ensure mathematical notation remains accurate.