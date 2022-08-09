# Liquid-Matrix-Segmentation
The problem of semantic segmentation is solved, namely the segmentation of a liquid matrix on a carbon fiber sample. 
An example of input data and masks are presented in the folder [data/imgs](https://github.com/MissDarya/Liquid-Matrix-Segmentation-/tree/main/data/imgs), [data/masks](https://github.com/MissDarya/Liquid-Matrix-Segmentation-/tree/main/data/masks), respectively

- To mark up the data, the following app was used https://github.com/hkchengrex/MiVOS
- To solve the segmentation problem, the Unet architecture was used, the implementation was borrowed from https://github.com/milesial/Pytorch-UNet

Examples of prediction on test dataset are shown: https://github.com/MissDarya/Liquid-Matrix-Segmentation-/tree/main/data/test/predict
 
**Dice Metric on Test DataSet is 0.93**
