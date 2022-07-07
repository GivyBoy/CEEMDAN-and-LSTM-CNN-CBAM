# Paper: `Forecasting gold price using a novel hybrid model with ICEEMDAN and LSTM-CNN-CBAM by Yanhui Liang, Yu Lin, and Qin Lu`

This repository holds the code for a paper I found interesting and decided to replicate. I was already interested in the Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN) algorithm, so this paper piqued my interest. I didn't use the Improved CEEMDAN (ICEEMDAN) algo, because I already had an implementation of the CEEMDAN algorithm that I, coincidentally, implemented a couple of days prior to finding this paper. It was fascinating implementing this project, since I have little to no deep learning experience. I learned a lot and I will definitely be exploring more deep learning methods.

I ensured that I document the code as best as possible, so persons can be able to follow along with ease. If you encounter any problems, feel free to reach out to me: `anthonygivans876@gmail.com`. If this repo interests you, feel free to check out my `quant_funcs` repo for more code like this. I also implemented the EMD and the EEMD algos. Thanks!

## Disclaimer:

Even though this algo fusion did really well in its forecasting, I highly doubt that it would work well in the real world. The CEEMDAN algo took the entire time series into account when it generated the Intrinsic Mode Functions (IMFs). That said, even though some powerful neural nets were thrown at the problem, there was definitely information leakage, which accounted for (some) the  predictive power. 