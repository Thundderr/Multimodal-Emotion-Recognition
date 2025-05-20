# CSC 566 - NEW AND IMPROVED MULTIOMODAL EMOTION RECOGNITION (tbd)

## Group Members
- Aiden Smith
- Abby Mercer
- Ethan Outangoun
- Scott Pramuk

## Instructions

1. Start up a venv, ideally with python 3.12.x
2. In the root directory run pip install -r requirements.txt
3. For some reason the model has an ungodly size so uh train your own in the python notebook found in root dir, you can run it for 5 epochs probably
4. Run python app.py
5. Go to your localhost at port 5000
6. Cry if it doesnt work or throws an error

## Links

- https://www.kaggle.com/datasets/sudarshanvaidya/random-images-for-face-emotion-recognition
- https://github.com/maelfabien/Multimodal-Emotion-Recognition


# For audio branch
1. Make a /Data directory
    - /raw subdirectory
2. download the video_speech_actor(x).zip from https://zenodo.org/records/1188976
3. unzip in the raw directory 
4. winget install ffmpeg if on windows
5. run 
    ```bash
        python extract_media.py
    ```
6. run 
    ```bash
        python audio_preprocessing.py
    ```
7. run 
    ```bash
        python train_audio_model.py
    ```