# sister-cloudmask
This repository contains a series of algorithms for cloud masking of hyperspectral imagery.


## CNN classifier

![](./examples/prisma_cnn.png)

The classifier is a two-step process:
1. A 1D CNN is used to perform and initial pixel-wise classifiction
2. A set of median and dilation filters are used to filter out stray pixels and buffer around clouds.

The models were trained using PRISMA (VNIR and VSWIR models) and DESIS (VNIR model) spaceborne radiance data.

## HyCMA classifier

![](./examples/prisma_hycma.png)

This classifier is based on the HyspIRI Cloud Mask Detection Algorithm (HyCMA) and uses a series
of thresholds on individual bands, indices and band ratios. Thresholds were adjusted to maximize
cloud detection accuracy in PRISMA imagery.


| Variable| HyCMA Threshold | PRISMA Threshold |
| ----------- | ----------- |----------- |
| 650nm  | 20 | 15.8 |
| NDSI  | 0.65 | 0.43|
| 800nm/650nm  | 4 | 4.4 |
| 800nm/550nm   | 2 | 1.4 |
| $\frac{\frac{800nm}{1650nm}}$	  | 1.5 | 2.3 |




HyspIRI Cloud Mask Detection Algorithm Theoretical Basis Document
Hulley, Glynn C.; Hook, Simon J.
[http://hdl.handle.net/2014/42573](http://hdl.handle.net/2014/42573)