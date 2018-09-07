Convolutional Recurrent Neural Network
======================================

This software implements the Convolutional Recurrent Neural Network (CRNN) in gluon.
Origin software could be found in [crnn](https://github.com/bgshih/crnn)

Run predict
--------
A demo program can be found in `predict.py`.

Edit the model_path and img_path in `predict.py`. Then launch the  `predict` by:

    python predict.py

The demo reads an example image and recognizes its text content.


Train a new model
-----------------
1. Create a file with image paths and labels

    ```sh
    datapath.jpg label
    datapath.jpg label
    datapath.jpg label
    ```

2. Create an alphabet in `keys.py` based on dataset 
 
3. modify the script `train.py` and run

    ```python
    python3 train.py
    ```

Dependence
----------
* mxnet 1.3.0
* mxboard
