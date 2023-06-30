# Hand Gesture Recognition using MediePipe

This is a simple program to detect three hand gestures, open palm, closed palm, and thumbs up (🖐️, ✊, 👍). The program can also be used to add new training data for gesture classifier and re-train the classifier to add more gesters.

![](https://github.com/AnandKumarRajpal/hand-gesture-recognition-using-mediapipe/blob/main/assets/Jun-30-2023%2022-40-21-compressed.gif?raw=true)

## Demo

Install the requirements:
```bash
pip install -r requirements.txt
```
Run the demo program:
```bash
python app.py
```

## Training

### Adding new data to the dataset:

To add new data press the `l` key when the app is running. You will now enter the data logging mode. Now press the keys from `0 - 9` while showing the required gesture. The gesture coordinates will be stored along with the pressed number. This pressed number represents the class of the store co-ordinates. More number of keys can be added based of the requirement. To return back to prediction mode press the `n` key.

### Training the classifier:

To train the classifier, open the `keypoint_classification.ipynb` file in Jupyter Notebook and run it from top to bottom.

We can add more classes to the classifier by changing the value of `NUM_CLASSES = 3` in the notebook and train the classifier.

After adding the new classes, we also need to modify the labels in `keypoint_classifier/keypoint_classifier_label.csv`.
