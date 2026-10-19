# Predicting Oral Temperature from Infrared Facial Measurements

**Authors:** Lei Huang and Shuyi Liu  
**Project:** Bayesian regression and random forest regression in R

[Report source](workflow_combined.qmd) · [HTML report](workflow_combined.html) · [Main test comparison](results/combined_comparison/main_model_comparison.csv)

## Research aim

We use existing volunteer measurements to simulate a temperature prediction workflow: facial infrared measurements and environmental conditions are the inputs, and oral reference temperature is the quantity to predict.

Our research question is: **How accurately can Bayesian regression and random forest regression predict oral temperature from these inputs?** We examine prediction errors and, for the Bayesian model, prediction uncertainty. FLIR and ICI data are analysed separately. The main comparison is between modelling approaches within each device.

This is a course project using an existing dataset. Its results describe predictive performance under the conditions represented in those data.

## Data

We use [*Facial and oral temperature data from a large set of human subject volunteers*, version 1.0.0](https://physionet.org/content/face-oral-temp-data/1.0.0/), available through PhysioNet under **CC0 1.0**. Dataset DOI: [10.13026/3bhc-9065](https://doi.org/10.13026/3bhc-9065).

The raw inputs are `data/FLIR_groups1and2.csv` and `data/ICI_groups1and2.csv`. The response is `aveOralM`, the mean of two oral monitor-mode measurements, saved as `target` in the analysis files. Temperatures are in degrees Celsius.

Each analysis row represents one subject for one device. We average each facial variable over its available measurement rounds, giving 26 facial predictors, plus ambient temperature, relative humidity and distance: **29 predictors** in total. The device datasets contain overlapping volunteers, so their row counts should be interpreted separately. Variable definitions are provided in Tables 1–2 on the dataset page and in the exported variable dictionary.

## Analysis workflow

1. Apply the shared cleaning rules, record corrections and missing values, and aggregate measurement rounds. The facial reading cutoff of 30°C is a project preprocessing convention.
2. Reuse the existing subject assignments in **`data/split_train_test_aveOralM.csv`**. Preserve this file when reproducing the analysis.
3. Develop the models using the training data and cross-validation grouped by acquisition date.
4. Fit the chosen specifications on the full training data and evaluate them on the same test subjects within each device.

The primary **Bayesian Gaussian regression** includes the 29 predictors and one quadratic term derived from `canthiMax`. Predictors are standardized using training statistics. Coefficient priors are Normal(0, 0.5); sensitivity checks use standard deviations 0.2 and 1.0. We examine MCMC diagnostics, posterior predictive checks, residuals and 90% prediction intervals.

The primary **random forest** uses recursive feature elimination, a one-standard-error selection rule, training cross-validation for tuning, and 1,000 trees in the final forest. An all-predictor forest, a training-mean baseline and simple linear regression provide reference comparisons.

Bayesian preprocessing is fitted within each CV analysis fold. The RF implementation retains the original approach of estimating medians on the full outer training set before inner CV. The two approaches also use different CV resamples; their development scores should be interpreted with these differences in mind.

## Model comparison

The main comparison uses **traditional test R², RMSE and MAE**, calculated from identical test outcomes for both models:

$$
R^2 = 1 - \frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y_{\mathrm{test}})^2},
\qquad
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_i(y_i-\hat y_i)^2}.
$$

Lower RMSE and MAE indicate smaller errors; higher R² indicates better prediction relative to the test-mean reference. Error measures and interval widths are in °C. Bayesian point predictions use posterior means of the expected response; intervals use the posterior predictive distribution.

The report's **full-training Bayesian R²** describes fitted-model variation. Prior sensitivity is evaluated separately by comparing predictions and performance across priors. The final model comparison uses the common test metrics in [main_model_comparison.csv](results/combined_comparison/main_model_comparison.csv). With identical test outcomes, lower RMSE also gives higher traditional R².

## Repository contents

| Path | Purpose |
|---|---|
| `workflow_combined.qmd` | Executable analysis with both model implementations |
| `workflow_combined.html` | Rendered report; download and open in a web browser |
| `data/` | Original device CSV files and the supplied subject split |
| `data/bayes_shared/` | Prepared datasets, saved split and `preparation.rds` |
| `models/bayes_shared/quadratic/` | Per-device `*_final_fit.rds` and `*_final_bundle.rds` |
| `models/rf_shared/` | Saved RF bundles; validated prediction exports can also be used |
| `results/bayes_shared/`, `results/rf_shared/` | CV results, diagnostics, preprocessing records and RF exports |
| `results/combined_comparison/` | Common test predictions, scores, intervals and input checksums |
| `results/combined_report/` | Tables and metadata exported by the report |
| `results/environment/shared/` | Recorded R, package and Quarto environment information |
| `report_assets/` | Figures and reported summaries used as fallbacks |
| `renv.lock`, `.Rprofile`, `renv/` | Package environment and activation files |

## Reproduce the analysis

Install the R version recorded in `renv.lock` and install [Quarto](https://quarto.org/) separately. The recorded Quarto version is in `results/environment/shared/quarto-version.txt`.

Open R in the project root, where `workflow_combined.qmd` is located, and restore the packages:

```r
source("renv/activate.R")
renv::restore()
renv::status()
```

Key packages include `brms`, `rstan`, `posterior`, `bayesplot`, `caret`, `ranger`, `doParallel`, `foreach`, `readr`, `dplyr`, `tidyr`, `ggplot2`, `knitr`, `rmarkdown`, `yaml` and `renv`. Restore the complete lockfile, including dependencies such as `shiny` inferred from the report parameters.

For the default run, keep the prepared data, saved Bayesian models, results and figures at their documented paths. If model files are distributed in an accompanying archive, extract it into the project root with these paths preserved. RF results can be supplied as bundles or as matching `*_test_predictions.csv` and `*_training_ids.csv` exports under `results/rf_shared/`.

Run in a terminal from the project root:

```bash
quarto render workflow_combined.qmd --to html
```

This reuses saved Bayesian models, trains RF models when their saved outputs are missing, and evaluates the fixed models on the supplied test split. Close output CSV files in Excel or WPS before rendering so they can be updated.

To repeat preparation and Bayesian fitting from the raw data:

```bash
quarto render workflow_combined.qmd --to html -P rebuild_models:true
```

This requires both raw CSV files and the original split file, runs 44 MCMC fits, and needs a compatible C++ toolchain for Stan, such as Rtools on Windows. Existing RF outputs are still reused when available. Choose model specifications using training results before the final test comparison.

## Reproducibility and FAIR

The project supports FAIR practices through the source DOI and version, documented file paths, CSV outputs with subject keys and units, a variable dictionary, explicit cleaning rules and priors, saved folds and seeds, input checksums, and a shared `renv.lock`. Environment records describe the software used for the combined workflow.

Raw data retain their **CC0 1.0** license; project code uses the **MIT License** in [LICENSE](LICENSE). Repository commits and releases identify analysis versions. If large data or model files are hosted separately, their download location and access instructions should accompany the repository.

Lei Huang contributed the Bayesian analysis; Shuyi Liu contributed the random forest analysis. The combined workflow aligns data preparation and test evaluation.

## Interpretation

Facial and oral temperature measure different quantities. Correlation motivates prediction, while test errors and interval coverage show how well the fitted relationship performs. The volunteer sample and acquisition conditions limit generalization to other populations and settings. These results support an educational modelling comparison; clinical use would require further validation.

## Dataset citation

Wang, Q., Zhou, Y., Ghassemi, P., Chenna, D., Chen, M., Casamento, J., Pfefer, J., and Mcbride, D. (2023). *Facial and oral temperature data from a large set of human subject volunteers* (version 1.0.0). PhysioNet. https://doi.org/10.13026/3bhc-9065.
