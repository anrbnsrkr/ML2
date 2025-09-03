Install requirements


torch==2.5.1
torchvision==0.20.1
tensorflow==2.18.0
keras==3.6.0
pandas==2.2.3
numpy==2.1.3
matplotlib==3.9.2
scikit-learn==1.5.2
joblib==1.4.2
opencv-python==4.10.0.84



Steps:
1. Place "dataset" folder containing part_one_dataset and part_two_dataset in the root folder.
2. Run feature_extractor.ipynb that will extract the features from raw dataset (D1 to D20) and store it in Augmented_Data directory. or download the augmented data directly from https://drive.google.com/file/d/1PRvH31WFJO9HfYw5QFPwGaUHOuhXBrIJ/view?usp=sharing (extracted and stored).
3. Run task1.ipynb file that will train the models f1 to f10, the final model f10 is saved as f10_model.pkl file for task 2.
4. Finally run task2.ipynb file that will train the models f11 to f20.
