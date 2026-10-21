# From Relational EHR Data to Longitudinal Patient Trajectories 

The software and data in this repository are part of a research study aimed at evaluating multiple embedding-based strategies for representing longitudinal clinical data to be used as input for supervised learning in downstream classification models. The generated embeddings were used jointly with other tabular
features derived from additional patient information. The classification task focused on mortality prediction in asplenic patients utilizing two patient cohorts: the INA real-world clinical dataset and an external validation dataset derived from MIMIC-IV. Deep learning models were used exclusively to generate patient-level temporal embeddings, while downstream classification was performed using a LightGBM model.

## Note

We included in this repository:
- the deidentified version of the original Italia Network for Aspleni (INA) dataset (`data/INA dataset.json`)
- the experimental pipeline conceived and implemented in a Jupyter notebook (`notebook.ipynb`), and
- the processing workflow, implemented in a Jupyter notebook (`mimic_to_json.ipynb`), for the extraction and representation in JSON format of longitudinal clinical trajectories of asplenic patients from MIMIC-IV relational database.
- all embedding models considered in this study are provided in the form of our implementations (`scripts/models.py`) wrote down by following the instructions in the related works

## Authors
- [Maurizio Giordano](https://orcid.org/0000-0001-9917-7591) and [Ilaria Granata](https://orcid.org/0000-0002-3450-4667)
- High Performance Computing and Networking (ICAR), Italian National Council of Research (CNR)

## Running the experiments
The experiments can be run individually by using the `notebook.ipynb` that can be execute in the Google Colab (or Jupyter Lab) environments.

The experiments can be run by terminal with the commands.


```
# Main experiments
python notebook.py --dataname INA --datafile INA_dataset_rev1.json \
  --results-dir experiment_main100 --mode embed --folds-dir-mimic folds100 --iter 10
python notebook.py --dataname MIMIC --datafile MIMIC_dataset_rev1.json \
  --results-dir experiment_main100 --mode embed --iter 10
# End-to-end experiments
python notebook.py --dataname INA --datafile INA_dataset_rev1.json --results-dir experiment_e2e100 --mode e2e --methods Med2Vec --methods Dipole --methods BiPadLSTM --methods GRU --iter 10
python notebook.py --dataname INA --datafile INA_dataset_rev1.json --results-dir experiment_embed00 --mode embed --methods Med2Vec --methods Dipole --methods BiPadLSTM --methods GRU --iter 10
# ICD code experiments
python notebook.py --dataname INA --datafile INA_dataset_rev1.json \
  --results-dir experiment_code100 --evfields code --mode embed --iter 10
# Shuffling experiments
python notebook.py --dataname INA --datafile INA_dataset_rev1.json \
  --results-dir experiment_shuffle100 --mode embed --iter 10 --shuffle
```

During running the script `notebook.pyn` produces several temporary files (in `--results-dir` forlder) collecting metric measures. These measure need to be processed by the `metric_calculations.ipynb` notebook to sum-up metric statistics
(mean, standard deviation, confidence  intervals, etc).
Once summary files are generated for metric values, you can plot the results by using one of the associate notebook:

- `results_main.ipynb` displays radar plots of results from Main experiments
- `stats_analysis.ipynb` performs statistical significance analysis of results from Main experiments 
- `calibration_analysis.ipynb` performs calibration analysis of results from Main experiments 
- `memory_analysis.ipynb` displays table of CPU, memory and time consumption of Main experiments 
- `pycarets.ipynb` perform comparison of LighGBM classifier vs other methods with PyCarte toolset 
- `results_main.ipynb` for Main experiments
- `results_ablation_e2e.ipynb` displays radar plots (and comparison barplots) for End-to-end experiments
- `results_ablation_code.ipynb` displays radar plots (and comparison barplots) for ICD code experiments
- `results_ablation_shuffle.ipynb` displays radar plots (and comparison barplots) for Shuffling experiments

The notebook `mimic_to_json.ipynb` implements the worflow for generating a unified JSON representation  of clinical data starting from relational database MIMIC-IV sources (you need a licence from Physionet to download MIMIC-IV v.3.1 source data)
