"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    n, l, h, w = infer_network.load_model(args.m, args.d, args.l)
    net_input_shape = self.plugin.get_input_shape()

    ### TODO: Handle the input stream ###
    image_flag = False
    # Check if the input is a webcam
    if args.i == 'CAM':
        args.i = 0
    elif args.i.endswith('.jpg') or args.i.endswith('.bmp'):
        image_flag = True

    ### TODO: Loop until stream is over ### 
    # Need to peer chat about this loop thing
    while cap.isOpened():
        # Read the next frame/image
        flag, image = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        # Break if escape key pressed
        if key_pressed == 27:
            break

        ### TODO: Read from the video capture ###
        cap = cv2.VideoCapture(args.i)
        cap.open(args.i)

        ### TODO: Pre-process the image as needed ###
        p_image = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
        p_image = p_image.transpose((2,0,1))
        p_image = p_image.reshape(1, *p_image.shape)

        ### TODO: Start asynchronous inference for specified request ###
        self.plugin.exec_net(p_image)

        ### TODO: Wait for the result ###
        while True:
            status=exec_net.requests[0].wait(-1)
            if status == 0:
                break
            else:
                time.sleep(1)

            ### TODO: Get the results of the inference request ###
            if self.plugin.wait() == 0:
                result = plugin.get_output()

            ### TODO: Extract any desired stats from the results ###
            def draw_boxes(image, result, args, width, height):
                '''
                Draw bounding boxes onto the image.
                '''
                for box in result[0][0]:
                    conf = box[2]
                    if conf >= args.pt:
                        xmin = int(box[3] * width)
                        ymin = int(box[4] * height)
                        xmax = int(box[5] * width)
                        ymax = int(box[6] * height)
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), 1)
                return image
            
            def draw_masks(result, width, height):
                '''
                Draw semantic mask classes onto the image.
                '''
                # Create a mask with color by class
                classes = cv2.resize(result[0].transpose((1,2,0)), (width,height),
                                    interpolation=cv2.INTER_NEAREST)
                unique_classes = np.unique(classes)
                out_mask = classes * (255/20)
                
                # Stack the mask so FFmpeg understands it
                out_mask = np.dstack((out_mask, out_mask, out_mask))
                out_mask = np.uint8(out_mask)
                
                return out_mask, unique_classes
                
            # Counter
            def counter(result, counter):
                # Need more time to peer chat on some parts of the project
                
                
            # Get the output of inference
            if self.plugin.wait() == 0:
                result = self.plugin.get_output()
                # Draw the output mask onto the input
                out_image, classes = draw_masks(result, width, height)
                

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            # Need to peer chat about calculating the relevant information
            client.publish("person", json.dumps({
                "count": count,
                "total": total
            }))
            client.publish("person/duration", json.dumps({"duration": duration}))

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(out_image)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        # Will peer chat about this
        cv2.imwrite(out_image)

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
