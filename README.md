# pascal3d

This branch:
- allows processing of 'imagenet' and 'pascal3d'
- creates 3D bounding box information and puts it in .npy
- outputs all images that fully contain the bounding box, by class

[![Build Status](https://travis-ci.org/wkentaro/pascal3d.svg?branch=master)](https://travis-ci.org/wkentaro/pascal3d)


Python version toolkit for [PASCAL3D](http://cvgl.stanford.edu/projects/pascal3d.html) dataset.  
The Matlab/Octave version is supported [in official](http://cvgl.stanford.edu/projects/pascal3d.html).  


## Install

```bash
./install.sh
```


## Usage

```bash
cd examples/dataset
./show_annotation.py
```

<img src="static/show_annotation.png" width="50%" />

```bash
./show_cad_ovelay.py
```

<img src="static/show_cad_overlay.png" width="50%" />

```bash
./show_cad.py
```

<img src="static/show_cad.png" width="50%" />

```bash
./show_depth.py
```

<img src="static/show_depth.png" width="50%" />
