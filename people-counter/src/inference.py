#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import logging as log

from openvino.inference_engine import IECore
import ngraph as ng


class Network:
    """
    Load and configure inference plugins for the specified target devices
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None
        self.device = None
        self.batch_size = 1
        self.infer_request_handle = None


    def get_supported_layers(self):
        """
        Get list of supported layers of the loaded network
        """
        ng_function = ng.function_from_cnn(self.network)
        return [node.get_friendly_name()
                for node in ng_function.get_ordered_ops()]

    def get_unsupported_layers(self):
        """
        Get the list of unsupported layers by the Network
        """
        supported_layers = (self.plugin.query_network(self.network, "CPU")).keys()

        layers = self.get_supported_layers()
        unsupported_layers = set(layers) - set(supported_layers)
        return list(unsupported_layers)


    def load_model(self, model, device="CPU", batch_size=1):
        """
        Load the model
        :param model: Model IR file
        :param device: device name
        :param batch_size: Batch size
        """
        self.device = device
        self.batch_size = batch_size

        model_structure = model
        model_weights = os.path.splitext(model)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()

        # Read the IR as a IENetwork
        self.network = self.plugin.read_network(model=model_structure, weights=model_weights)
        self.network.batch_size = self.batch_size

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(network=self.network, device_name=self.device)

        # Check the supported layers of the network
        unsupported_layers = self.get_unsupported_layers()
        if unsupported_layers:
            str_layers = ', '.join(unsupported_layers)
            log.error(f"Following layers are not supported by "
                        "the plugin for the specified device {self.device}:\n {str_layers}")
            sys.exit(1)

        # Get the input layer
        self.input_blob = next(iter(self.network.input_info))

        self.output_blob = next(iter(self.network.outputs))
        return self.network, self.get_input_shape()


    def get_input_shape(self):
        """ Return the shape of the input layer
        """
        # log.info(f"Input INFO {self.network.input_info}")
        return self.network.input_info[
            list(self.network.input_info.keys())[0]
        ].input_data.shape


    def exec_net(self, image, request_id=0):
        """
        Start an asynchronous inference request, given an input batch of images.
        :param batch: the batch of images
        :return : The handle of the asynchronous request
        """
        self.infer_request_handle = self.exec_network.start_async(
            request_id=request_id, inputs={self.input_blob: image}
        )
        return self.infer_request_handle


    def wait(self, timeout=-1):
        """
        Wait for an asynchronous request
        :return outputs:
        A dictionary that maps output layer names to
        numpy.ndarray objects with output data of the layer.
        """
        while self.infer_request_handle.wait(timeout=timeout) != 0:
            log.info("Waiting for inference ...")
            time.sleep(2)
        return self.infer_request_handle.output_blobs[self.output_blob]


    def get_output(self):
        """
        Extract the output results
        :return
           Inference request result
        """
        return self.infer_request_handle.output_blobs[self.output_blob].buffer

