# PeopleCounterApp
An AI at the Edge app 

In this project, you will utilize the Intel® Distribution of the OpenVINO™ Toolkit to build a People Counter app, including performing inference on an input video, 
extracting and analyzing the output data, then sending that data to a server. The model will be deployed on the edge, such that only data on 
1) the number of people in the frame, 2) time those people spent in frame, and 
3) the total number of people counted are sent to a MQTT server; inference will be done on the local machine.


Provided Files
The project has files implementing:
1.	A MQTT server - receives JSON from your primary code subsequent to inference concerning people counted, duration they spent in frame, and total people counted.
This will feed to the UI server.
2.	A UI server - displays a video feed as well as statistics received from the MQTT server.
3.	A FFmpeg server - receives output image frames including any detected outputs inferred by the model. This is then fed to the UI server.


A video file is also provided to test the implementation of code in People Counter app.


Code for implementation is split into two files:
1.	inference.py - Here, you will load the Intermediate Representation of the model, and work with the Inference Engine to actually perform inference on an input.
2.	main.py - Here, you will:
•	Connect to the MQTT server
•	Handle the input stream
•	Use your work from inference.py to perform inference
•	Extract the model output, and draw any necessary information on the frame (bounding boxes, semantic masks, etc.)
•	Perform analysis on the output to determine the number of people in frame, time spent in frame, and the total number of people counted
•	Send statistics to the MQTT server
•	Send processed frame to FFmpeg


Hardware and software requirements
•	64-bit operating system that has 6th or newer generation of Intel processor running either Windows 10, Ubuntu 18.04.3 LTS, or macOS 10.13 or higher.
•	Installing OpenVINO (version 2020.1) on your local environment. Its following link for instructions on how to install: https://docs.openvinotoolkit.org/2020.1/index.html
•	Installing Intel’s Deep Learning Workbench (version 2020.1) Please note that DL Workbench does not currently support Windows 10 Home Edition. 
It is recommended that you either upgrade to Windows 10 Professional or use a Linux based system. 
Link for installation instructions: https://docs.openvinotoolkit.org/2020.1/_docs_Workbench_DG_Install_Workbench.html
•	Installing Intel’s VTune Amplifier: https://software.intel.com/en-us/get-started-with-vtune
Loading pre-trained models
Pre-trained models allow you to explore your options without needing your own data or training a model.
They are fed directly to the inference engine, as these are in Intermediate Representation (IR) format. I am using one of Intel Edge AI nanodegree’s pre-trained 
models as a demonstration of how to load a pre-trained model. 
On your local machine where you installed openvino toolkit, navigate to the directory containing the model downloader:
cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
e.g, downloading the human pose model:
sudo ./downloader.py --name human-pose-estimation-0001 -0 /home/workspace

--name for model name, and --precisions, used when only certain precisions are desired.

Verifying Downloads
The downloader itself will tell you the directories these get saved into, but to verify yourself, first start in the same directory as the Model Downloader on 
your local machine. From there, you can cd intel, and then you should see three directories (by typing ls in cmd) – one for each downloaded model.
Within those directories, there should be separate subdirectories for the precisions that were downloaded, and then .xml and .bin files within those subdirectories,
that make up the model.

The Model Optimizer

first download the SSD MobileNet V2 COCO model by writing this in cmd:
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
Then use the tar -xvf command with the downloaded file to unpack it.
Documentation to converting TensorFlow’s Object Detection Zoo is:
https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html

If the conversion is successful, the terminal should let you know that it generated an IR model. The locations of the .xml and .bin files, as well as
execution time of the Model Optimizer, will also be output.

The Inference Engine
Runs the actual inference on a model. It provides hardware-based optimizations to get even further improvements from a model.
There are two types of inference requests:
Synchronous: will wait and do nothing else until the inference response is returned, blocking the main thread.
Asynchronous: when the response for an item takes a long time, you don’t hold up the rest of your website or app from loading or operating appropriately. 
It means other tasks may continue while waiting on the IE to respond.
We will be using asynchronous requests in the codes.

Running the App
Sourcing the Environment
When opening a new terminal window, in order to source the environment, use the command:
Source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
Any new terminals you open will again need this command run.

From the main directory:
Step 1 – Start the Mosca server
cd webservice/server/node-server
node ./server.js
You should see the following message if successful:
Mosca server started.

Step 2 – Start the GUI
Open new terminal and run below commands:
cd webservice/ui
npm run dev
You should see the following message in the terminal:
Webpack: compiled successfully

Step 3 – FFmpeg Server
Open new terminals and run the below commands:
sudo ffserver -f ./ffmpeg/server.conf

Step 4 – Run the code
Open a new terminal to run the code.
Running on the CPU
When running Intel® Distribution of OpenVINO™ toolkit Python applications on the CPU, the CPU extension library is required. This can be found at:
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/
Depending on whether you are using Linux or Mac, the filename will be either libcpu_extension_sse4.so or libcpu_extension.dylib, respectively. 
(The Linux filename may be different if you are using a AVX architecture)

Though by default application runs on CPU, this can also be explicitly specified by -d CPU command-line argument:

python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m your-model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

While running locally, to see the output on a web-based interface, open the link http://0.0.0.0:3004 in a browser.
