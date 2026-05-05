# A Multimedia Player Controlled by Imagined Movements via BCI

## Project Overview
This project combines Brain-Computer Interface (BCI) technology with multimedia player controls. It allows users to control multimedia playback through imagined movements, offering a hands-free interaction experience.

## Key Features
- Control multimedia players using BCI technology.
- Real-time response to imagined movements.
- User-friendly interface for configuration and usage.

## Repository Structure
- **src/**: Contains the source code for the BCI control logic.
- **models/**: Includes pre-trained models for movement classification.
- **data/**: Datasets used for training the models and input data expectations.
- **config/**: Configuration files for setting up the project environment.
- **README.md**: This documentation file.
- **LICENSE**: License details regarding the usage of the code.

## Prerequisites
- Python 3.6 or later
- Required packages listed in `requirements.txt`
- A compatible OS (Windows, macOS, or Linux)

## Installation
1. Clone the repository:  
   `git clone https://github.com/Premsivasai/A-Multimedia-Player-Controlled-by-Imagined-Movements-via-BCI.git`
2. Navigate to the project directory:  
   `cd A-Multimedia-Player-Controlled-by-Imagined-Movements-via-BCI`
3. Install required packages:  
   `pip install -r requirements.txt`

## Usage
To train the model, use the following command:  
`python src/train_model.py --data_path data/ --output_path models/`

To run inference or real-time control, use:  
`python src/run_control.py --model_path models/your_model.h5`

## Configuration
Configuration files located in the `config/` directory can be modified to suit your needs, including the choice of dataset and model parameters.

## Dataset / Input Expectations
Input data should match the structure and format defined in the documentation within the `data/` folder. Ensure all datasets are properly pre-processed before training.

## Model Details
The project utilizes machine learning models trained to classify imagined movements from BCI/EEG signals, focusing on recognizing user intent for multimedia control.

## Troubleshooting
- Ensure all prerequisites are installed.
- Verify EEG/BCI device connectivity (if applicable).
- Review command parameters and paths.

## License
No license specified.
