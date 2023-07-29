# Computer Vision Pipeline Example

This project is an example of a computer vision pipeline.

* **Goal**: to detect the coordinates of the nose of a person using the webcam
* **Summary**:
    1) **Dataset**: a dataset containing photos and the corresponding nose coordinates was assembled. A labeling tool was developed for this step and it can be executed through `run_labeler.sh`
    2) **Model training**: a pre-trained convolutional neural network was fine-tuned with the dataset and images obtained from data-augmentation
    3) **Inference**: the model is used in real-time, detecting coordinates from a webcam video stream. Inference can be executed from `run_inference.sh`

## Setup
```
pip install requirements.txt
```

## Utilities
A set of utility functions and classes were developed for this project and they can be found at the `common` folder.

## Labeling Tool
The labeling tool uses inputs from the user to create bounding boxes for each unlabeled image. Although the goal of this project is to detect point coordinates, this tool was can be used for a broader range of applications. It works as follows:
* the tool loops through all images located at `data/unlabeled`
* for each image, the user must click twice in order to define the **upper left** and the **bottom right** coordinates that define the rectangle over the object of interest
* the tool will display the original image with the rectangle. The user can accept the label with a **right click**
* the accepted label is stored in a *json* file in `data/labels`
* the labeled image is moved to `data/images`

The tool can be executed by sourcing `run_inference.sh`.

## Model Training
The model training procedure is run at `training/train.ipynb`. A brief description is provided below.

### Dataset
A dataset consisting of 99 image/label pairs was build using the labeling tool. Then, a random selection was done in order to split the dataset into the following subsets:
* **training subset**: 49 samples
* **validation subset**: 25 samples
* **test subset**: 25 samples

Every image was pre-processed using the `RawImageProcessor` object that resized the images to a fixed (250, 250) shape and converted them to grayscale.

### Augmentation
* The [Albumentations](https://albumentations.ai/) library was used to augment the images and the labels (or keypoints)
* The pipeline consisted of a `RandomResizedCrop` followed by a `HorizontalFlip` transformation
* This was only applied to the training subset before the training loop, such that the 49 original samples were expanded to 294 samples

### Data Loaders
* A `Dataset` object was built for each subset, which can provide pairs of image and label in the correct format that the model requires
* A `ModelImageProcessor` object was used, which is reponsible of tranforming the input image to a `torch.Tensor` object with the correct normalization (intensities in the [0,1] range) and correct tensor shape
* The training, validation and test datasets were used by a `DataLoader`, responsible of providing batches of data to the model

### Training
* The chosen model was a `MobileNet-V2`, which was pre-trained and can be downloaded from the [PyTorch repositories](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)
* The original model was altered in order to consider a 2D output, representing the normalized coordinates with respect to the image shape. (a `Sigmoid` activation function was included for normalization)
* A custom loss function was developed (sum of coordinates absolute errors) and it can be found at `training/loss.py`
* The training loop was executed for 100 epochs with the use of a GPU
* The model was serialized with `pickle` and stored at `training/saved_models`


## Inference

The following description explains how inference is performed:
* A `Runner` object is used, which executes the following methods:
    * `initialize_capture()`: initializes webcam video stream and logs when it starts or fails
    * `run()`: runs the inference loop:
        * firstly, it attempts to capture an image frame. If it fails, the module tries again after 3 seconds for a maximum of 2 times, raising an error if it cannot get the frame
        * then, the following model pipeline is executed in the frame:[`RawImageProcessor`, `ModelImageProcessor`, model call]
        * the center of the frame is computed and a rectangle is determined, where its edges are $10 \%$ of the original dimensions in $X$ and $Y$
        * if the detected coordinate is within the rectangle and if the previous coordinate wasn't, an `Event` is created and this is logged
        * a list of events is stored in the `Runner` object and incremented as new events are detected
        * **the running loop can be stopped by pressing `q`**
    * `stop_capture()`: stops webcam video stream once the previous step ends

* The steps of instantiating and running the `Runner` object is done in a separate thread
* The number of detected events is printed at the console
* Logs are stored at `out.log`
