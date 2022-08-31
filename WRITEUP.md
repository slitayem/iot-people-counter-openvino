# Project Write-Up

This project was made in the context of an assignment for the `Intel(R) Edge AI for IoT Developers Nanodegree` [program by Udacity](https://www.udacity.com/course/intel-edge-ai-for-iot-developers-nanodegree--nd131). 

The goal of the project is to find a useful person detection model and convert it to an Intermediate Representation to use with the Model Optimizer. Utilizing the Inference Engine, I use the model to perform inference on an input video, and extract useful data concerning the count of people in frame and how long they stay in frame. This information get sent over MQTT, as well as sending the output frame, in order to view it from a separate UI server over a network.

# Table of contents
- [Used setup](#used-setup)
- [Custom Layers in OpenVINO™](#custom_layers)
- [Comparing Models Performances](#models_testing)
- [Selected model](#best_model)
- [Model use cases](#usecases)
- [Effects on end user needs](#usecase_effects)
- [Further reading](#further-reading)


Below I explain some concepts related with OpenVINO Toolkit usage as well as the model selection for the people counter app.

<a name="#used_setup"></a>

# Used setup

## Software
    - Ubuntu 20.04.3 LTS in Docker image
    - Intel® Distribution of OpenVINO™ toolkit v2021.4.752
    - Node v16.16.0
    - npm v8.15.0
    - MQTT Mosca server 2.8.3
    - CMake 3.21.4

## Hardware
- Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz

Note: You can get the CPU model name using the following command
```bash
lscpu | grep "Model name:" | sed -r 's/Model name:\s{1,}//g'
```

<a name="#custom_layers"></a>

# Custom Layers in OpenVINO™

> The `Model Optimizer` is a cross-platform command-line tool facilitating the transition between the training and deployment environments. It performs static model analysis, and adjusts deep learning models for optimal execution on end-point target devices. It then produces an `Intermediate Representation (IR)` of the network, which can be inferred with the Inference Engine.

The IR of the network is then a pair of files describing the model:
- `.xml` Describes the network topology
- `.bin` Contains the weights and biases binary data.

<img src="https://docs.openvino.ai/2021.4/_images/BASIC_FLOW_MO_simplified.svg"
    caption="architectural diagram" width="500" class="center">

Before performing that process, it searches for each layer of the input model in the list of known layers.
Chances are that the model requires some operations which are not in the list of known layers for each of the supported [frameworks (Tensorflow, ONNX, Caffe, etc)](https://docs.openvino.ai/2021.4/openvino_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html) (e.g especial activation functions, loss functions, etc). Those layers are considered as a [custom layers](https://docs.openvino.ai/2021.4/openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html#doxid-openvino-docs-m-o-d-g-prepare-model-customize-model-optimizer-customize-model-optimizer).

### Performance Implications
Creating custom layers can create performance issues in certain conditions (see [Deep Learning Inference Engine Workflow](https://www.intel.com/content/www/us/en/developer/articles/technical/openvino-custom-layers-support-in-inference-engine.html))

Custom kernels are used to quickly implement missing layers for cutting-edge topologies. For that reason, it is not advised to override layers on the critical path (for example, Convolutions). Also, overriding existing layers `can disable some existing performance optimizations such as fusing`.

When the custom layers are at the very `end of the pipeline`, it is easier to implement them as regular post-processing in the application without wrapping as kernels. This is particularly true for kernels that do not fit the GPU well, for example, (output) bounding boxes sorting. In many cases, one can do such post-processing on the CPU.

<a name="#models_testing"></a>

# Comparing Models Performance

Several models were tested to find out the most convenient one for the use case application. Pre-trained [Public](https://docs.openvino.ai/2021.4/omz_models_group_public.html) and [Intel](https://docs.openvino.ai/2021.4/omz_models_group_intel.html) models were used. Those are listed below with their respective complexity and accuracy details.

| Model Name | Implementation | Complexity (GFLOPs)  | Size(MParams) | Accuracy (%) | Input Shape [BxCxHxW]|
-------------------------------------------------------------------------------------------------------------------------------- |----|--------------------- |----------- |------ |-------- |
[ssd_mobilenet_v2_coco](https://docs.openvino.ai/2021.4/omz_models_model_ssd_mobilenet_v2_coco.html) | Tensorflow|3.775 | 16.818 | (coco precision) 24.9452% | [1x3x300x300]|
[pedestrian-detection-adas-0002](https://docs.openvino.ai/2021.4/omz_models_model_pedestrian_detection_adas_0002.html) |	Caffe|2.836 | 1.165 | AP 88% |  [1x3x384x672] |
[person-detection-retail-0013](https://docs.openvino.ai/2021.4/omz_models_model_person_detection_retail_0013.html)|Caffe|2.300 | 0.723 | AP 88.62%| [1x3x320x544]|
[ssdlite_mobilenet_v2](https://docs.openvino.ai/2021.4/omz_models_model_ssdlite_mobilenet_v2.html)| TensorFlow| 1.525| 4.475 |24.2946%| [1x3x480x864]|

A `confidence detection thershold` of 80% was used for the person objects detection.

The models were converted to `IR format` in different `floating-point precisions` (FP16 and FP32).

The average inference time as well as the completion time to process the whole video is reported below.

| Model Name                     |   Detected People count |   AVG inference time (sec.) |   Completion Time (Sec.) | Floating-point precision   |
|:-------------------------------|---------------:|----------------------------:|-------------------------:|:---------------------------|
| ssdlite_mobilenet_v2           |             26 |                       0.015 |                  123.607 | FP16                       |
| pedestrian-detection-adas-0002 |             27 |                       0.02  |                  131.595 | FP16                       |
| ssdlite_mobilenet_v2           |             27 |                       0.016 |                  127.325 | FP32                       |
| pedestrian-detection-adas-0002 |             27 |                       0.025 |                  141.195 | FP32                       |
| person-detection-retail-0013   |             30 |                       0.022 |                  137.503 | FP32                       |
| person-detection-retail-0013   |             31 |                       0.017 |                  128.227 | FP16                       |
| ssd_mobilenet_v2_coco          |             32 |                       0.021 |                  132.175 | FP16                       |
| ssd_mobilenet_v2_coco          |             32 |                       0.022 |                  134.186 | FP32                       |

<a name="#best_model"></a>

# Selected model

Considering the size of the model, number of detected people and average inference time, the selected model for the project is the public pre-trained model `person-detection-retail-0013` with half-precision floating point numbers precision ([FP16](https://www.intel.com/content/www/us/en/developer/articles/technical/should-i-choose-fp16-or-fp32-for-my-deep-learning-model.html)).

<a name="#usecases"></a>

# Model use cases
Some of the potential use cases of the people counter app are:
- Prevent unauthorized access to buildings or particual factory areas
- Help to guarantee that a place is not overcrowded to avoid the spread of COVID-19 virus.
- Keep track of the average duration of time spend by customers in different store locations. This can help in understanding the customers preferences in the store.

<a name="#usecase_effects"></a>

# Effects on end user needs
Depending on the end user use case the accuracy need as well as the video capture conditions would differ. Factors impacting the detection results would be `lighting, camera focal length, image resolution, position of the camera`.
For example, a low lighting in the video capture environment can affect the objects detection algorithm ability to detect the people.

In the case of a usecase setup having low capacity in terms of hardware resources, we would need to make a trade-off between the model accuracy and the inference time.

# Further readings
- [Pre-trained models](https://docs.openvino.ai/latest/omz_models_group_intel.html)
- [OpenVINO model downloader](https://docs.openvinotoolkit.org/latest/omz_tools_downloader.html)
- [Custom Layers in the Model Optimizer](https://docs.openvino.ai/2019_R3/_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html)
- [Model Optimizer Extensibility](https://docs.openvino.ai/2021.4/openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html)
- [Choose FP16, FP32 or int8 for Deep Learning Models](https://www.intel.com/content/www/us/en/developer/articles/technical/should-i-choose-fp16-or-fp32-for-my-deep-learning-model.html)
- [OpenVINO `2021.4.2` documentation](https://docs.openvino.ai/2021.4/index.html)
- [Custom Layers Support in Inference Engine](https://www.intel.com/content/www/us/en/developer/articles/technical/openvino-custom-layers-support-in-inference-engine.html)
- [Evaluation Metrics for Object Detection](https://blog.zenggyu.com/en/post/2018-12-16/an-introduction-to-evaluation-metrics-for-object-detection/)s




