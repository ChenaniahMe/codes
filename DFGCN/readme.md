#### INTRODUCTIOIN
The code includes three parts: feature extraction, graph construction, and network model.

#### Data
The original data contains T.mat, label.mat, and feature extraction needs to be performed on T.mat

#### RUN CODE
##### FEATURE EXTRACTION
You need to run the file DealData/SmallFlevoland/matlab/main.m first to extract feature, using T.mat and label.mat.
```bash
run main.m
```
we can get Fea_V.mat. 

##### GRAPH CONSTRUCTION
We copy both Fea_V.mat, label.mat, T.mat to DealData/SmallFlevoland/python/

You need to run the file DealData/SmallFlevoland/python/main.py second to extract feature.
```bash
python main.py
```

we can get Feature.npy, label.npy, index.npy， adj.npy

#### NETWORK MODEL
The training of the model code is under the TrainCode/DFGCN4
You need to modify config.py, SmallFlevoland/python/ corresponding label.npy
file_name = "SmallFlevoland"
You need to modify utils.py, SmallFlevoland/python/ corresponding Feature.npy, label.npy, adj.npy
file_name = "SmallFlevoland/python/"
Lastly, You need to run the file DFGCN4/train.py to train model.

#### VISUALIZATION
The visualization of the data is under the TrainCode/DFGCN4
You need to modify picture.py to load label.mat
file_name = "SmallFlevoland"
You need to run the file picture.py to visualize data results. To modify the parameter draw_picture is 0 to get the original image, and to modify the parameter draw_picture is  to get the test result image


#### REQUIREMENTS
look requirements.txt 
cuda version： cuda 9.0
operation system： ubuntu1~16.04.10