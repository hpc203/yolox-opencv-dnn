# yolox-opencv-dnn
使用OpenCV部署YOLOX，支持YOLOX-S、YOLOX-M、YOLOX-L、YOLOX-X、YOLOX-Darknet53五种结构，包含C++和Python两种版本的程序

onnx文件在百度云盘，下载链接：https://pan.baidu.com/s/11UAVSPWbDKY_LmmoHlUQXw 
提取码：147w

下载完成后，把文件放在代码文件所在目录里，就可以运行程序了。如果出现读取onnx文件失败，
那很有可能是你的opencv版本低了，需要升级到4.5以上的


在10月20日，我看了一下官方代码https://github.com/Megvii-BaseDetection/YOLOX
新版的在做推理时，预处理没有做BGR2RGB，除以255.0, 减均值除以方差这几步的。
因此如果用最新代码训练后生成onnx文件，然后用本仓库里的程序做推理时，需要注释掉“BGR2RGB，
除以255.0, 减均值除以方差这几步”
