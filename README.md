# Installation guide
1. Clone repository:

    ```
    git clone https://github.com/HiGal/People-Tracking.git
    ```
2. Cd to the cloned folder and install requirements

    ```
    cd path/to/People-Tracking/
    pip install -r requirements.txt
    ```    

3. Download weights of the model into **People-Tracking/cfg/**

    ```
    wget https://pjreddie.com/media/files/yolov3.weights
    ```

4. Run on the video:

    ```
    python yolo_video.py --input input/<your-video> --output output/<name of your video> --yolo cfg
    ```
