# Udacity Capstone Project - Perception Module

# Introduction
This project is part of the Udacity Capstone Project with the aim of programming a real self-driving car. Our team created several ROS nodes to implement core functionality of a self-driving car including perception, path-planning and control system design. In this repository, I'll talk about the perception module in this project.For more information regarding the Capstone project and the our solution please visti the following two links: [Udacity github](https://github.com/udacity/CarND-Capstone) and [our team solution](https://github.com/fstahl1/CarND-Capstone). 

# Traffic Light Detection
The mission of perception module in the Capstone project was detection of the traffic lights and classification of its state while driving in the simulator or in the real test track. So, an object dection model trained using images of traffic lights in two different environment (simulator and test track) is required. There was no obstacles either in the simulator nor on the test site. Hence, the obstacle detection was not explored in our peception module. Having checked different models and their trained datasets in [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection), it was found that they can detect traffic light in an image without specifing the state of the light. So, initially I thought about two different approaches:
- Using one of these models for detecting the traffic lights, masking the image to keep only the lights in its detected bounding box and then sending the masked image to a classifier to determine state of the traffic light.
- Retraining one of those models for end-to-end detection which can classify the entire image as containing either a red light, yellow light, green light, or no light.

The first approach seems to be easier to be implemented. However, it consists of two different modules working sequentially which might adversly effect on the process speed. Therefore, retraing the pretrained models in Tensorflow Object Detection API using new datasets was considered as an appropriate approach for this project. As stated in the requirements of the Udacity simulator, the trained model should be compatible with tensorflow 1.3.0. I had Windows 10 and installed tensorflow 1.14 on my local machine which I didn't want to downgrade. So, I retrained my model with this tensorflow and then exported it to a comptible version with the simulator. The following steps have been taken to accomplish this goal:
- Setting up the environment
- Image collection, labeling data and producing TFrecord
- Retraing the tensorflow object detection model using our dataset
- Performance evaluation of the new model in the integrated system
- Exporting the new model to Tensorflow 1.4 (comptible version with Udacity simulator)

Reading [Alex Lechner's githaub](https://github.com/alex-lechner/Traffic-Light-Classification) and [Sendtex's tutorial](https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/) have been extremely helpful in all steps.

## 1. Environment Set up - Windows 10
Tensorflow Object Detection API depends on the following libraries: Jupyter notebook, Python, Tensorflow (>=1.12.0), Protobuf, Pillow, lxml, tf Slim, Matplotlib, Cython, contextlib2. 
Having [Anaconda](https://docs.anaconda.com/anaconda/install/windows/) and Python (also in the PATH environment variable):
1. [Install Tensorflow 1.14 with GPU support](...my readme on tensorflow gpu) 
2. Install the following python packages
```
pip install pillow lxml matplotlib Cython contextlib2
```
3. Clone TensorFlow's models repository in a new dircetory (named tensorfolow) on your local machine by executing
```
git clone https://github.com/tensorflow/models.git
```
4. Download protoc-3.11.1-win64.zip (compatible version with tensorflow 1.14) from the [Protobuf repository](https://github.com/protocolbuffers/protobuf/releases)
5. Extract the Protobuf .zip file to e.g. C:\Program Files\protoc-3.11.1-win64
6. Navigate to your "...\tensorflow\models\research" directory and execute the following command in the cmd (replace "..." with your path):
```
"Path_to_Your_protoc-3.11.1-win64_Folder "/protoc-3.11.1-win64/bin/protoc object_detection/protos/*.proto --python_out=.
```
7. In order to access the modules from the research folder from anywhere, the models, models/research, models/research/slim & models/research/object_detection folders need to be set as PATH variable ([see the full instruction here](https://stackoverflow.com/questions/48247921/tensorflow-object-detection-api-on-windows-error-modulenotfounderror-no-modu)) 

## 2. Preparing The Dataset
Since the simulator and the real test track environments are quiet different, we should also use two distinct datasets due to differences in the appearance of the traffic lights in these two environment.

### 2.1. Dataset From Udacity Simulator
For this section, I collected some data from the simulator and labled them using LableImg. In order to get good performance from the model, I needed decent amount of data. So, I also used [Alex's simulator datases](https://github.com/alex-lechner/Traffic-Light-Classification#1-the-lazy-approach) and [Vatsal's dataset](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset). Alex's data and my own data together were combined and used for training. Vatsals dataset was used for evaluation purpose. 
##### Labeling The Images
For labaling the data, first we need to [download LableIm](https://github.com/tzutalin/labelImg). To install lableImg, open the Anaconda Prompt and go to the labelImg directory (cloned github)
```
conda install pyqt=5
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py
```
Then, everytime you want to lable new sets of image, in order to open lableImg just open the Anaconda Prompt and go to the labelImg directory (C:\Udacity_Final_Project\Dataset\LableImg ToolBox\labelImg-master) and:
```
python labelImg.py
```
Then:
- Click on Open Dir and select the folder of your traffic lights images
- Create a new folder within the traffic lights folder and name it labels
- In labelImg click on Change Save Dir and choose the newly created labels folder. Now you can start labeling your images. When you have labeled an image with a bounding box hit the Save button and the program will create a .xml file with a link to your labeled image and the coordinates of the bounding boxes ([Source: Alex Lechner's github](https://github.com/alex-lechner/Traffic-Light-Classification#1-the-lazy-approach)).
As recommended, the traffic light images were splitted into 4 folders: Green, Red, Yellow and Background. The advantage is that you can check Use default label and use e.g. Red as an input for your red traffic light images and the program will automatically choose Red as your label for your drawn bounding boxes 

##### Creating TFRecord
After labeling the images, we should make a label_map.pbtxt file which contains our labels (Green, Red, Yellow & Background) with an ID (IDs must start at 1). Then, a TFRecord (a binary file format which stores our images and ground truth annotations) was created to be used in retraining the TensorFlow model. This was done in create_tf_record.py script. For datasets with .xml files execute (replace "..." with your paths):
```
python create_tf_record.py --data_dir="Path_To_Your_GreenLightImages"/Green,"Path_To_Your_RedLightImages"/Red,"Path_To_Your_YellowLightImages"/Yellow --annotations_dir=labels --output_path="Path_To_Destination_Folder"/MyTrain_dataset.record --label_map_path="Path_To_Your_LabelMap_Folder"/label_map.pbtxt
```
And, now MyTrain_dataset.record is generated. This process should be also repeated for the evaluation datasets. 

### 2.2. Test Track Dataset
For the sake of retraining the perception model for test track, I used the images provided in udacity-rosbag-file and labled them with labelImg. Again, having enough samples was a concern. So, I combined my recorded data with [Alex's site dataset](https://github.com/alex-lechner/Traffic-Light-Classification#1-the-lazy-approach) for training and used [Vatsal's site dataset](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset) for evaluation purpose. TFrecords were created in the end using these two datasets.

## 3. Training the Traffic Light Classification Model
- Download a model from the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and extract the file. Here, I have used the ssd_mobilenet_v1_coco_2018_01_28 for simulator and ssdlite_mobilenet_v2_coco_2018_05_09 for the site,which both seem to be fast ([read more](https://arxiv.org/abs/1611.10012) and [find the useful source codes](https://github.com/udacity/CarND-Object-Detection-Lab)).

- Configure the pipeline.config file by making the following changes in:
```
	num_classes: 4
	max_detections_per_class: 10
	max_total_detections: 10
	fine_tune_checkpoint: "Path_To_Above_Extracted_Folder/model.ckpt"
	num_steps: 25000
	num_examples: 36
	label_map_path: "Path_to_Your_lableMap_Folder/label.pbtxt"	
	train_input_reader {input_path: ...} and eval_input_reader {input_path:... } to your train and evaluation TfRecord paths
```

- Make a new directory for saving the new model (named here NewTrain) and train the model based on the chosen model (replace the paths in "..." and run the following command in cmd) 
```
python "Path_to_tensorflow/models/research"/object_detection/model_main.py  --logtostderr   --pipeline_config_path="Path_to_Your_ConfigFile"/pipeline.config    --model_dir="Path_to_Your_New_Directory"/NewTrain/   
```
The steps start at 1 and the loss will be much higher (somthing between 10-30). Depending on our GPU and the size of training data, this process will take varying amounts of time. Our desirable loss is something about ~1 on average (or lower). 

## 4. Tensorboard Visulization
Progress of the retraing the model can be checked via TensorBoard during the training. Also, we can visualize the retraining process when it finished. First, you need to install the tensorboard, if it is not intalled yet:
```
pip install tensorboard
```
Then, run the following command (replace "..." with your path):
```
tensorboard --logdir="Path_to_Your_NewTrain_Folder"
```
Then simply open [http://localhost:6006](http://localhost:6006) in your browser to see TensorBoard. Here, is how my model has progress over time during the training:
<figure>
	<figcaption>Total loss - Retrained Model for the Simulator</figcaption>
	<img src="https://github.com/FElhamAsadi/CarND-TrafficLightDetection/blob/master/Results/Loss_total_loss.svg" width="800" 
</figure>
And sample outputs (right image is the ground truth):
![Sample Output_1](/Results/eval_1.png)

![Sample Output_2](/Results/eval_2.png)

## 5. Make the Tensorflow Graph From The Retraining Results 
When training the model finishes, you will have the following files in the NewTrain folder: 
```
checkpoint
graph.pbtxt
model.ckpt-NUMBER.data
model.ckpt-NUMBER.index
model.ckpt-NUMBER.meta
```
These files will be used for makeing the exportable tensorflow model. As mentioned before, we also should care about compatibility with the Udacity's simulator and Carla.  However, following instrucation is provided for two different cases: 

- Case 1: If there was no compatibility concerns at this point
- Case 2: If you need to export the trained model from tensorflow 1.14 to tensorflow 1.4

### 5.1. Case 1
Redirect to the NewTrain folder and run the command below (replace your paths in "..."):
```
python "Path_to_tensorflow/models/research"/object_detection/export_inference_graph.py     --input_type image_tensor      --pipeline_config_path "Path_to_Your_ConfigFile"/pipeline.config  --trained_checkpoint_prefix  model.ckpt-25000  --output_directory NewModel
```
Note 25000 was the number of steps that I chose for retraining the model. This output would be the graph as frozen_inference_graph.pb. Results should apper in the NewModel folder created the NewTrain. Now, we can use this frozen file for object detection in the tl_classifier.py.

### 5.2. Case 2
For exporting the trained model from tensorflow 1.14 to tensorflow 1.4, I recommend using linux (or VM-linux). Here, I took advantage of the Udacity provided VM to genereate my final tensorflow graph. First, you need to copy the NewTrian folder and your pipeline.config file to the VM. Then, follow these steps:

- Since anaconda was not installed on the Udacity's VM, first I used curl to download  and then install it:
```
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
sha256sum Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
```
- Install corresponding environments:
```
conda create -n tensorflow_1.4 python=2.7 
conda activate tensorflow_1.4 
pip install tensorflow==1.4.0 
conda install pillow lxml matplotlib 
```
- Here, we also need the object detection models for tensorflow 1.4 which can be found in the following githubs:
```
git clone https://github.com/tensorflow/models.git TempFolder 
cd TempFolder  
git checkout d135ed9c04bc9c60ea58f493559e60bc7673beb7 
cd .. 

mkdir exporter
cp -r TempFolder/research/object_detection exporter/object_detection 
cp -r TempFolder/research/slim exporter/slim 
cd TempFolder/research

git clone https://github.com/cocodataset/cocoapi.git 
cd cocoapi/PythonAPI 
make 
cd protobuf-3.4.0/ 
conda activate tensorflow_1.4 
```

- Install the corresponding version of Protocol buffer
```
wget "https://github.com/protocolbuffers/protobuf/releases/download/v3.4.0/protoc-3.4.0-linux-x86_64.zip" 
unzip protoc-3.4.0-linux-x86_64.zip "/home/student/exporter/" 
mv protoc-3.4.0-linux-x86_64.zip exporter/ 

cd exporter/ 
protoc object_detection/protos/*.proto --python_out=.   
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim 
```

- Export the model in tensorflow 1.4
```
mkdir /home/student/NewModel/   
EXPORT_DIR="/home/student/NewModel/" 
TRAINED_CKPT_PREFIX=/home/student/NewTrain/model.ckpt-25000
PIPELINE_CONFIG_PATH=/home/student/NewTrain/pipeline.config
INPUT_TYPE=image_tensor 
python /home/student/TempFolder/research/object_detection/export_inference_graph.py     --input_type=${INPUT_TYPE}     --pipeline_config_path=${PIPELINE_CONFIG_PATH}     --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX}     --output_directory=${EXPORT_DIR} 
```
The output should be found oin the NewModel folder as tf graph and can be used in tf_classifier.py.  
