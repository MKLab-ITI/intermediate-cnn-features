# Intermediate CNN Features
This repository contains the implementation of the feature extraction process described in
[Near-Duplicate Video Retrieval by Aggregating Intermediate CNN Layers](https://link.springer.com/chapter/10.1007/978-3-319-51811-4_21). 
Given an input video, one frame per second is sampled and its visual descriptor is extracted 
from the activations of the intermediate convolution layers of a pre-trained Convolutional Neural Network.
Then, the Maximum Activation of Convolutions (MAC) function is applied on the activation of each layer
to generate a compact layer vector. Finally, the layer vector are concatenated to generate a single
frame descriptor. The feature extraction process is depicted in the following figure.

<img src="https://raw.githubusercontent.com/MKLab-ITI/intermediate-cnn-features/develop/feature_extraction.png" width="60%">

## Prerequisites
* Python
* Caffe or Tensorflow 1.xx

## Getting started

### Installation

* Clone this repo:
```bash
git clone https://github.com/MKLab-ITI/intermediate-cnn-features
cd intermediate-cnn-features
```
* Install all dependencies:
```bash
pip install -r requirements.txt
```

### Feature Extraction

* Provide a file of videos or images to the corresponding argument. Each line 
of the given file have to contain the full path to one video or image.

* Select one of the two supported frameworks to extract features: Caffe or Tensorflow.

* Choose one of the three supported CNN architectures based on the selected framework. 
CNN architectures for each framework:
    * Caffe: GoogleNet, VGG, ResNet
    * Tensorflow: VGG, ResNet, Inception
    
* Provide the required files for the pre-trained networks. Downloads for
[Caffe](https://github.com/BVLC/caffe/wiki/Model-Zoo) and 
[Tensorflow](https://github.com/tensorflow/models/tree/master/research/slim).

* Provide an output path. The generated files depends on the input file.
    * Video list: a file is generated for each video with name ```<video_basename>_<selected_network>.npy```.
    * Image list: a file with name ```<selected_network>_features.npy``` is generated. Each vector in 
    the 0 axis of the stored numpy array correspond to the image in the input file.
* Caffe example:
```bash
python feature_extraction.py --video_list <video_file> --network googlenet --framework caffe --output_path test/ --prototxt bvlc_googlenet/deploy.prototxt --caffemodel bvlc_googlenet/bvlc_googlenet.caffemodel
```

* Tensorflow example:
```bash
python feature_extraction.py --image_list <image_file> --network vgg --framework tensorflow --output_path test/ --tf_model slim/vgg_16.ckpt
```

## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{kordopatis2017near,
  title={Near-Duplicate Video Retrieval by Aggregating Intermediate CNN Layers},
  author={Kordopatis-Zilos, Giorgos and Papadopoulos, Symeon and Patras, Ioannis and Kompatsiaris, Yiannis},
  booktitle={International Conference on Multimedia Modeling},
  year={2017}
}

```
## Related Projects
**[ViSiL](https://github.com/MKLab-ITI/visil)** **[NDVR-DML](https://github.com/MKLab-ITI/ndvr-dml)** **[FIVR-200K](https://github.com/MKLab-ITI/FIVR-200K)**

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details

## Contact for further details about the project

Giorgos Kordopatis-Zilos (georgekordopatis@iti.gr) <br>
Symeon Papadopoulos (papadop@iti.gr)
