# GRU-ow
Gated Recurrent Network (GRU) to GRUow your savings

## Overview
  * [Intro](#intro)
  * [Requirements](#requirements)
  * [Usage](#usage)
  * [Results](#results)
  * [Dex Swap](#Dex Swap)
  * [References](#references)


### Intro
Recent years have seen remarkable progress in the development of statistical and machine learning methods. Increasingly, developer/datascience time spent feature engineering is being replaced by compute cycles and growing data ingestion pipelines. Performance on sequence modeling tasks has improved dramatically with the introduction of deep neural netowrks such as Long-Short Term Memory (LSTM), Gated Recurrent Unit (GRU), and Attention based Transformer Networks. These models represent some of the most powerful dynamic memory autoregressive techniques, a quality essential in attempts to model the complex nonstationary (constantly changing) dynamics of asset exchange markets, such as that of ETH/USDC. In this repo we implement a single-hidden layer Gated Recurrent Network trained to forecast minute level price candles (OHLC), 8 minutes into the future, with the input of the most recent 128 minute candle+volume (OHLCV). Applications of this type of model extend to **volatitlity modeling for risk-management**, price impact of **trade execution optimization**, and model-based reinforcement learning **optimal portfolio construction**.


#### Architecture
The architectural design of the GRU circuit is illustrated below. Each GRU is designed with learnable weight parameters that determine the rate/level of information propogation across a sequence of observed isntances. These units effectively gate the memory of our model.
![gru_circuit](images/GRU_circuit.png)

Below is the GRU circuit's math, i.e. the above circuit in the form of math equations whose parameters we will train to estimate

![gur_maths](images/gru_maths.png)


### Requirements
  * Linux distributions that use glibc >= v2.17
  * Python 3.6
  * matplotlib == 3.1.1
  * numpy == 1.19.4
  * pandas == 0.25.1
  * scikit_learn == 0.21.3
  * torch == 1.8.0
  * mplfinance == 0.12.9b0
  * scipy == 1.7.3

Dependencies can be installed using the following command:
```
pip install -r requirements.txt
```
Optional - For training on a GPU (highly recommended), Nvidia CUDA 10.0+ drivers are required

### Usage

In the command line

```python
python -i model/train_model.py
```

![traing_loss](images/train_loss.png)



### Dex Swap



### Results

![predicitons_gif](images/animated_graph2.gif)



### References
