## DoG & JBF

> Difference of Gaussians (DoG) is a feature enhancement algorithm that involves the subtraction of one Gaussian blurred version of an original image from another, less blurred version of the original.[^1]
>
> Joint bilateral filter (JBF) is **a non-linear, edge-preserving, and noise-reducing smoothing filter for images**. It replaces the intensity of each pixel with a weighted average of intensity values from nearby pixels.[^2]
> [^1]: https://en.wikipedia.org/wiki/Difference_of_Gaussians
> [^2]: https://en.wikipedia.org/wiki/Bilateral_filter

In this project, we implement both DoG and JBF from scratch with OOP patterns. Part1 involves steps of creating gaussian images and DoG images, and eventually detecting feature points by the subtraction of DoG imgags. Part2 involves steps of computing spatial kernel and range kernel, and conducting joint bilateral filter with guidance image (grayscale). By evaluating the cost between JBF image and BF image, we can find the optimized conversion to grayscale image.


### Usage (Part1)

Plot keypoints with different `--threshold` on the `--image_path` by using DoG.

`python3 main.py --threshold 5.0 --image_path './testdata/1.png'`

### Usage (Part2)

Plot JBF image by using original image in `--image_path` and guidance image accoriding to the config in the `--setting_path`.

`python3 main.py --image_path './testdata/1.png' --setting_path './testdata/1_setting.txt'`

### Directory layout

    .
    ├── part1 (DoG)
    │   ├── DoG.py              # class of DoG
    │   ├── eval.py             # evaluation
    │   ├── main.py             # main program to run
    │   └── testdata
    │       ├── 1.png
    │       ├── 1_gt.npy
    │       └── 2.png
    └── part2 (JBF)
       ├── JBF.py              # class of JBF
       ├── eval.py             # evaluation         
       ├── main.py             # main program to run
       └── testdata
           ├── 1.png
           ├── 1_setting.txt
           ├── 2.png
           ├── 2_setting.txt
           ├── ex.png
           ├── ex_gt_bf.png
           └── ex_gt_jbf.png