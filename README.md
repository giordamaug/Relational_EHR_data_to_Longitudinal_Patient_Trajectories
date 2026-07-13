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

