# README Size Estimation Project
Tool for extracting frames from video files produced with Azure Kinect cameras. 

* [GRAP-UdL-AT/ka_frame_extractor](https://github.com/GRAP-UdL-AT/ka_frame_extractor)

## Installing Linux (TODO)
```
python3 -m venv ./ka_frame_extractor-venv
source ./ka_frame_extractor-venv/bin/activate
pip install --upgrade pip
pip install -r requirements_linux.txt
```
https://github.com/GRAP-UdL-AT/ka_frame_extractor
![alt text](https://github.com/GRAP-UdL-AT/ka_frame_extractor/blob/main/img/screen_linux.png?raw=true)


## Installing in Windows 10
From command line CMD
```
%userprofile%"\AppData\Local\Programs\Python\Python38\python.exe" -m venv ./venv
size_estimation-venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements_win.txt
```

![alt text](https://github.com/GRAP-UdL-AT/ka_frame_extractor/blob/main/img/screen_win.png?raw=true)

## Basic requirements
This software need the following packages:
```
pip install windows-curses
pip install pyk4a
pip install path
pip install opencv-python
```

## Authorship
This project is contributed by [GRAP-UdL-AT](http://www.grap.udl.cat/en/index.html).

Please contact authors to report bugs juancarlos.miranda@udl.cat

## Citation

If you find this code useful, please consider citing:
[GRAP-UdL-AT/ka_frame_extractor](https://github.com/GRAP-UdL-AT/ka_frame_extractor/).