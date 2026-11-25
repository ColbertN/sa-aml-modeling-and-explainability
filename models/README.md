# Models

Trained model binaries and preprocessing artefacts are not committed by default. They can be generated locally by the pipeline scripts and should be stored in an experiment artefact store with the dataset release, code commit, feature list, threshold, calibration method, and random seed.

The reusable artefacts include the fitted encoder/scaler, feature-name list, sequence configuration, model weights, classification threshold, and calibration parameters. Loading these artefacts is required for inference on unseen data; refitting preprocessing objects on future data would invalidate the evaluation.
