ConvertToCoreML
=====================================================

This repository contains resources for converting caffe or keras models to CoreML
models (.mlmodel)
------------------------------------------------------

Convertable Models
------------------
- MobileNet

Installation
-------------
Enter the directory of the model you wish to convert, and run

```
virtualenv -p /usr/bin/python2.7 env
source env/bin/activate
pip install tensorflow
pip install keras==1.2.2
pip install coremltools

python convert<ModelName>.py

deactivate # when you are done
```
