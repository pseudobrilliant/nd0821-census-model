# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Production model contained in the `./model` directory is a random forest classifier trained on the census data stored in `./data`.
The model comes packaged with a fitted encoder and label binarizer for categorical data while continous variables are unchanged.
The final production model selected was trained with the following model_parameters:
  - n_estimators: 500
  - criterion: gini
  - max_depth: 30
  - min_samples_split: 2
  - min_samples_leaf: 1
  - min_weight_fraction_leaf: 0.0
  - max_features: sqrt
  - max_leaf_nodes: null
  - min_impurity_decrease: 0.0
  - bootstrap: true
  - oob_score: false
  - n_jobs: null
  - random_state: 42
  - verbose: 0
  - warm_start: false
  - class_weight: balanced
  - ccp_alpha: 0.0
  - max_samples: null

## Intended Use

This model is intended to be used for the prediction of income classes based on census data.
  - age: int
  - workclass: str
  - fnlgt: int
  - education: str
  - education_num: int
  - marital_status: str
  - occupation: str
  - relationship: str
  - race: str
  - sex: str
  - capital_gain: int
  - capital_loss: int
  - hours_per_week: int
  - native_country: str

The following fields will be treated as categorical values to be one-hot-encoded.
  - workclass
  - education
  - marital_status
  - occupation
  - relationship
  - race
  - sex
  - native_country

The production model saved will generate a prediction as to whether the census item provided represents an individual with a salary of <=50K or >50K a year.

## Training Data
All models were trained on 22793 values of the training split (70%) from the 32561 items in the census dataset provided (`./data/raw`). The data was manually edited to address some white space, and naming convention issues (`./data/cleaned`)

## Evaluation Data
Models were evaluated using the remaining 9768 testing items (30%) from the total 32561 items provided in the dataset.
All model data, parameters, and artifacts were tracked using ml flow. This allowed for easy comparison and evaluation across the generated models.
This model was selected through a hydra experimentation cycle applying the following sweeps to the following parameters:
  - max_depth: 10,15,30,50,100,300
  - n_estimators: 50,100,200,500

## Metrics
The final production model was selected according to it's high beta score and balanced precision / recall (shown below).
  - fbeta: 0.688
  - precision: 0.6663
  - recall: 0.711

## Ethical Considerations
The data provided has been throughly cleaned of any identifying markers and is publicly accessible to all.
Therefore we are aware of no ethical concerns or unintended ethical implications at this time.

## Caveats and Recommendations
Other classification based models should be attempted for comparison. Additional parameter sweeps could also result in better performing models.
