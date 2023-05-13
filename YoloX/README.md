# Survey on YoloX

[Introduction](#introduction)  
[Data](#data)  
[Preparing the enviroment](#preparing-the-enviroment)  
[Testing the models](#testing-the-models)  
[Conclusion](#conclusion)  


# Introduction 

This is a survey conducted by Ahmed Hossam Fadl on the YoloX research paper found [here](https://arxiv.org/pdf/2107.08430.pdf), it tests the common models mentioned in the paper on the COCO dataset to try and replicate the results as well as give general ideas and thoughts on the pros and cons of YoloX versus it's predecessors Yolov3 and Yolov5.

# Data

The dataset used for this is the validation subset of the 2017 Version of the COCO dataset, the images can be found [here](http://images.cocodataset.org/zips/val2017.zip) and there annotations can be found [here](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

# Preparing the enviroment 

To prepare and enviroment for testing the repo, I simply cloned the repo using this link
```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
```
Then while inside the repo directory run the command 
```bash
pip3 install -v -e .
```
This should install all the required dependencies needed to run YoloX.

After that make a link inside the YoloX repo to path of your desired dataset
```bash
ln -s /path/to/your/COCO ./datasets/COCO
```

# Testing the models

This demonstrates how to test the YoloX-L version of YoloX but the same can be applied to any other model version by simply swapping the weights

First get the final weights for the model that can be found here [here](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth)

After that simply run the commaned 
```bash
python -m yolox.tools.eval -n  yolox-l -c /path/to/your/model/weights/yolox_l.pth -b 64 -d 1 --conf 0.001 [--fp16] [--fuse]
```

Assuming everything runs smoothly the output should be something like this 
![Output](./Output.png)

These results matches the [results mentioned in the original repo of YoloX](https://github.com/Megvii-BaseDetection/YOLOX#standard-models) for the YoloX-L version of the model

**Note that the time is a bit slower than the original repo because their testings are done using a V100 Tesla GPU which is much faster**

# Conclusion

The main point of strength for YoloX over it's predecessors is the fact that it's anchor-free meaning there is no need to pre-define anchor boxes before training, this makes it much easier to re-train/adapt the models on different domains and datasets without having to study the data to determine the suitable anchor boxes that will yield the best result. 

While YoloX does infact provide better results than it's predecessors taking a look at this result table from the paper 
![YoloX_VS_YoloV5](./YoloX_VS_YoloV5.png)

It seems that as the models get larger the gain in AP decreases until it reaches less than 1%, while the latency, model size and computations required are higher than it's predecessors at all model versions.

It is also worth noting that the newest models of Yolo ( [YoloV7](https://github.com/WongKinYiu/yolov7#performance) ) already beats YoloX in both performance and accuracy.

However this does not take away the fact that anchor-free training is a powerful advantage for YoloX that saves training time as well as provide a decent gain over YoloV3 and YoloV5 especially in smaller model versions of YOLO.
