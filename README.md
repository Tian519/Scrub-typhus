# Scrub typhus risk modeling in China

This repository contains R scripts for random forest modeling, SHAP analysis, and future projections of scrub typhus cases. 
Run the scripts in the following order:

01_fit_RF_models.R | Prepare the historical county-month data and fit 20 random forest models. 
02_RF_SHAP.R | Use the fitted models to calculate SHAP values and assess predictor contributions. 
03_ST_future_Predict.R | Combine the fitted models with future input data to predict county-month scrub typhus cases. 

Update the input and output paths before running each script. 
The restricted scrub typhus surveillance data are not included in this repository. Researchers seeking access should contact the Chinese Center for Disease Control and Prevention (China CDC).
