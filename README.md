# Real-Time Sign Language Recognition (MNIST)

This project uses a Convolutional Neural Network (CNN) to recognize American Sign Language (ASL) alphabet signs from a webcam feed. The model is trained on the Sign MNIST dataset.

## How to Run
1.  Clone the repository and install dependencies:
    ```bash
    git clone https://github.com/PriyanshAg-1/sign-language-recognition.git
    cd sign-language-recognition
    pip install -r requirements.txt
    ```

2.  Run the application:
    ```bash
    python run.py
    ```
The script will load the pre-trained model from the `models/` directory and start the webcam. If the model is not found, it will automatically train a new one using the data in `data/`, which can take a while.
