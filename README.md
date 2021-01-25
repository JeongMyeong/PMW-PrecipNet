
`Precipitation retrieval based on ConvolutionalNeural Networks from passive microwave observations`

# PrecipNet

PMW-PrecipNet was trained to retrieve instantaneous precipitation rate using three years products between 2016 and 2018 from Global Precipitation Measurement (GPM) core satellite, where the GPM Dual Precipitation Radar (DPR) rainfall rate is used as a reference. 

The results show that our proposed algorithm provides 19 \% of the improved correlation with Dual Precipitation Radar compared to the operational GPM precipitation retrieval algorithm.

## Model Architecture

- Base : U-Net
- First Layer : Add Downsampling
- Last Layer : Add Upsampling


![model](/assets/model.png)


## Training

- Requirements
  + Python 3
  + Tensorflow / Keras

- Data
  + 2016 ~ 2018 GPM data
  
- Method
  + Cross Validation
  + Ensemble
  
- Score
  + F1 score
  + MAE
  
## Citation

```
```
