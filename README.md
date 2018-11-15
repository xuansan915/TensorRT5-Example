# About This Sample
This sample demonstrates how to first train a model using TensorFlow and Keras, freeze the model and write it to a protobuf file, convert it to UFF, and finally run inference using TensorRT5.

# Installing Prerequisites
1. Make sure you have the python dependencies installed.
    - For python2, run `python2 -m pip install -r requirements.txt` from the top-level of this sample.
    - For python3, run `python3 -m pip install -r requirements.txt` from the top-level of this sample.
2. Make sure you have the UFF toolkit as well as `graphsurgeon` installed. if not, download it from [here](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing-tar)

3. Train the model and write out the frozen graph:
    ```
    $ mkdir models
    $ python train.py
    ```
# Create a TensorRT inference engine

## use frozen graph
1. Convert the .pb file to .uff, using the convert-to-uff utility:
    ```
    $ convert-to-uff ./models/model.pb -o ./models/
    ```
    The converter will display information about the input and output nodes, which you can use to the register
    inputs and outputs with the parser. In this case, we already know the details of the input and output nodes
    and have included them in the sample.

2. Create a TensorRT inference engine from the uff file and run inference:
    ```
    $ python uff2plan.py
    ```
    The data directory needs to be specified only if TensorRT is not installed in the default location.
    
## use uff file
1. Convert the .pb file to .uff, using the convert-to-uff utility:
    ```
    $ convert-to-uff models/model.pb
    ```
    The converter will display information about the input and output nodes, which you can use to the register
    inputs and outputs with the parser. In this case, we already know the details of the input and output nodes
    and have included them in the sample.

2. Create a TensorRT inference engine from the uff file and run inference:
    ```
    $ python uff2plan.py
    ```
    The data directory needs to be specified only if TensorRT is not installed in the default location.
