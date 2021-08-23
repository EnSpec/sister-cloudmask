# sister-cloudmask
This is a simple classification algorithm for generating a 4 class mask (land, water, snow/ice and cloud) from
radiance data.

The classifier is a two-step process:
1. A 1D CNN is used to perform and initial pixel-wise classifiction
2. A set of median and dilation filters are used to filter out stray pixels and buffer around clouds.

The models were trained using PRISMA (VNIR and VSWIR models) DESIS (VNIR model) spaceborne radiance data.

## Installation

```bash
pip -r requirements.txt
```

## Use

```bash
python cloudmask.py radiance_image output_directory
```

Optional arguments:

- `--verbose`: default = False
- `--median`: Size of median filter, default = 7
- `--dilation`: Size of dilation filter, default = 7
- `--apply`: Create a copy of the input radiance image and set cloud pixels to 'no data' value, default = False

