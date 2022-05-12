# Food-detection

This repo contains the code for training and testing 2D food segmentation models. The link for the dataset and the models are provided below. I have used detectron2 library for training and inference. So, you first need to download the detectron2 library in your local machine. GPU is preferred for training and inference. You also need to download the cuda and cudnn. Detectron2 is based on pytorch so it's better to use the pytorch version which detectron2 is using.

You can run the notebook `train.ipynb` for training the model. You need to specify the datapath in your local machine after the installation process. The `dataset.ipynb` notebook contains the code for preparing the dataset in coco format. I have already done this so it doesn't need to be run. For testing the model, you can used `test.py` file. You need to specify the trained model path and the testing image path inside the file. 

```
Link for dataset and model: https://drive.google.com/drive/folders/1m1BUic3ay83nhl_FsU3zmPTUxfaXtCpm?usp=sharing
```

You need to download the dataset, unzip and place it into your local system. The datset is divided into test and train with their respective annotatations in COCO format. I have already trained one model named 'model.pth' for maskrcnn_resnet_101. You can also used that directly in test.py. This also contains the results inferenced on our trained model.

Before testing, please install all the library in `requirements.txt`