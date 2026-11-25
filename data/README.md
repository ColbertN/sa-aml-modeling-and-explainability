# Data

The complete synthetic dataset is maintained in the companion repository and archived in Zenodo:

<https://github.com/ColbertN/SA-AML-DATA-SIMULATION>

<https://zenodo.org/records/22803333>

It was created because South African AML transaction histories are not generally available for reproducible academic research. The simulator uses publicly documented South African financial-crime patterns and locally relevant banking, currency, payment, grant, corridor, customer, and typology attributes.

Only the schema is versioned in this repository. The raw CSV is intentionally excluded because it is large and should be obtained from the companion data project or the Zenodo release. The Zenodo record should be used to identify the exact immutable data release used in an experiment. Before running a pipeline:

1. Obtain the data release from the companion repository.
2. Confirm the columns against `raw/SA_SAML_D_schema.json`.
3. Record the simulator release or commit used for the experiment.
4. Keep the raw file outside Git unless a dedicated data-governance decision authorises distribution.

The modelling code expects transaction labels for evaluation, but target, scenario, and simulator-internal fields are excluded from predictive inputs.
