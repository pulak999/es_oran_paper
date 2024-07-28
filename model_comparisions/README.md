# Forecasting Model Comparision
## By Pulak Mehrotra

- We compared the performance of many industry leading forecasting methods (ARIMA, Prophet and LSTM) with varying types of time-series data.
- Time-series data in use:
    - Microsoft stock prices over time ==> Microsoft_Data
    - Temperature change over a time period ==> Temperature_Data
    - Power Readings ==> COMED_Data
- Each folder has an implementation of each of the above mentioned forecasting methods. 
    - The ARIMA and Prophet methods are retrained everytime they are run, since they are memoryless.
    - The LSTM models saved in a given folder have been trained on the dataset associated with that folder. They do not need to be retrained. The training portion can be ignored, simply load the model.

For any clarifications, please contact me directly at +91 9956872143 or f20210017@pilani.bits-pilani.ac.in.