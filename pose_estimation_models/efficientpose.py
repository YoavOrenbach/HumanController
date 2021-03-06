from tensorflow.keras.applications.imagenet_utils import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.backend import set_learning_phase
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation
from tensorflow.keras.backend import sigmoid, constant
from tensorflow.keras.initializers import Initializer
import numpy as np
import math
from skimage.transform import rescale
from skimage.util import pad as padding
from scipy.ndimage.filters import gaussian_filter
import os
from pose_estimation_models.pose_estimation_logic import PoseEstimationLogic


class EfficientPose(PoseEstimationLogic):
    def __init__(self):
        super(EfficientPose, self).__init__("efficientpose")
        self.model, self.resolution, self.lite = None, None, None
        self.sub_model = 'II_Lite'

    def load_model(self):
        self.model, self.resolution, self.lite = get_model(self.sub_model)

    def start(self):
        pass

    def process_frame(self, frame):
        coordinates = analyze(frame, self.model, self.resolution, self.lite)
        landmarks = coordinates_to_landmarks(coordinates)
        return landmarks

    def end(self):
        pass

    def get_landmark_names(self):
        landmark_names = ['head_top', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax',
                          'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee',
                          'right_ankle', 'left_hip', 'left_knee', 'left_ankle']
        return landmark_names


class Swish(Activation):
    """
    Custom Swish activation function for Keras.
    """

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'Swish'


def swish1(x):
    """
    Standard Swish activation.

    Args:
        x: Keras tensor
            Input tensor

    Returns:
        Output tensor of Swish transformation.
    """

    return x * sigmoid(x)

def eswish(x):
    """
    E-swish activation with Beta value of 1.25.

    Args:
        x: Keras tensor
            Input tensor

    Returns:
        Output tensor of E-swish transformation.
    """

    beta = 1.25
    return beta * x * sigmoid(x)


class keras_BilinearWeights(Initializer):
    """
    A Keras implementation of bilinear weights by Joel Kronander (https://github.com/tensorlayer/tensorlayer/issues/53)
    """

    def __init__(self, shape=None, dtype=None):
        self.shape = shape
        self.dtype = dtype

    def __call__(self, shape=None, dtype=None):

        # Initialize parameters
        if shape:
            self.shape = shape
        self.dtype = type=np.float32 # Overwrites argument

        scale = 2
        filter_size = self.shape[0]
        num_channels = self.shape[2]

        # Create bilinear weights
        bilinear_kernel = np.zeros([filter_size, filter_size], dtype=self.dtype)
        scale_factor = (filter_size + 1) // 2
        if filter_size % 2 == 1:
            center = scale_factor - 1
        else:
            center = scale_factor - 0.5
        for x in range(filter_size):
            for y in range(filter_size):
                bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * \
                                       (1 - abs(y - center) / scale_factor)

        # Assign weights
        weights = np.zeros((filter_size, filter_size, num_channels, num_channels))
        for i in range(num_channels):
            weights[:, :, i, i] = bilinear_kernel

        return constant(value=weights)

    def get_config(self):
        return {'shape': self.shape}


def resize(source_array, target_height, target_width):
    """
    Resizes an image or image-like Numpy array to be no larger than (target_height, target_width) or (target_height, target_width, c).

    Args:
        source_array: ndarray
            Numpy array of shape (h, w) or (h, w, 3)
        target_height: int
            Desired maximum height
        target_width: int
            Desired maximum width

    Returns:
        Resized Numpy array.
    """

    # Get height and width of source array
    source_height, source_width = source_array.shape[:2]

    # Compute correct scale for resizing operation
    target_ratio = target_height / target_width
    source_ratio = source_height / source_width
    if target_ratio > source_ratio:
        scale = target_width / source_width
    else:
        scale = target_height / source_height

    # Perform rescaling
    resized_array = rescale(source_array, scale, multichannel=True)

    return resized_array


def pad(source_array, target_height, target_width):
    """
    Pads an image or image-like Numpy array with zeros to fit the target-size.

    Args:
        source_array: ndarray
            Numpy array of shape (h, w) or (h, w, 3)
        target_height: int
            Height of padded image
        target_width: int
            Width of padded image

    Returns:
        Zero-padded Numpy array of shape (target_height, target_width) or (target_height, target_width, c).
    """

    # Get height and width of source array
    source_height, source_width = source_array.shape[:2]

    # Ensure array is resized properly
    if (source_height > target_height) or (source_width > target_width):
        source_array = resize(source_array, target_height, target_width)
        source_height, source_width = source_array.shape[:2]

    # Compute padding variables
    pad_left = int((target_width - source_width) / 2)
    pad_top = int((target_height - source_height) / 2)
    pad_right = int(target_width - source_width - pad_left)
    pad_bottom = int(target_height - source_height - pad_top)
    paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]
    has_channels_dim = len(source_array.shape) == 3
    if has_channels_dim:
        paddings.append([0,0])

    # Perform padding
    target_array = padding(source_array, paddings, 'constant')

    return target_array


def preprocess(batch, resolution, lite=False):
    """
    Preprocess Numpy array according to model preferences.

    Args:
        batch: ndarray
            Numpy array of shape (n, h, w, 3)
        resolution: int
            Input height and width of model to utilize
        lite: boolean
            Defines if EfficientPose Lite model is used

    Returns:
        Preprocessed Numpy array of shape (n, resolution, resolution, 3).
    """

    # Resize frames according to side
    batch = [resize(frame, resolution, resolution) for frame in batch]

    # Pad frames in batch to form quadratic input
    batch = [pad(frame, resolution, resolution) for frame in batch]

    # Convert from normalized pixels to RGB absolute values
    batch = [np.uint8(255 * frame) for frame in batch]

    # Construct Numpy array from batch
    batch = np.asarray(batch)

    # Preprocess images in batch
    if lite:
        batch = efficientnet_preprocess_input(batch, mode='tf')
    else:
        batch = efficientnet_preprocess_input(batch, mode='torch')

    return batch

def extract_coordinates(frame_output, frame_height, frame_width, real_time=False):
    """
    Extract coordinates from supplied confidence maps.

    Args:
        frame_output: ndarray
            Numpy array of shape (h, w, c)
        frame_height: int
            Height of relevant frame
        frame_width: int
            Width of relevant frame
        real-time: boolean
            Defines if processing is performed in real-time

    Returns:
        List of predicted coordinates for all c body parts in the frame the outputs are computed from.
    """

    # Define body parts
    body_parts = ['head_top', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle']

    # Define confidence level
    confidence = 0.3

    # Fetch output resolution
    output_height, output_width = frame_output.shape[0:2]

    # Initialize coordinates
    frame_coords = []

    # Iterate over body parts
    for i in range(frame_output.shape[-1]):

        # Find peak point
        conf = frame_output[...,i]
        if not real_time:
            conf = gaussian_filter(conf, sigma=1.)
        max_index = np.argmax(conf)
        peak_y = float(math.floor(max_index / output_width))
        peak_x = max_index % output_width

        # Verify confidence
        if real_time and conf[int(peak_y),int(peak_x)] < confidence:
            peak_x = -0.5
            peak_y = -0.5
        else:
            peak_x += 0.5
            peak_y += 0.5

        # Normalize coordinates
        peak_x /= output_width
        peak_y /= output_height

        # Convert to original aspect ratio
        if frame_width > frame_height:
            norm_padding = (frame_width - frame_height) / (2 * frame_width)
            peak_y = (peak_y - norm_padding) / (1.0 - (2 * norm_padding))
            peak_y = -0.5 / output_height if peak_y < 0.0 else peak_y
            peak_y = 1.0 if peak_y > 1.0 else peak_y
        elif frame_width < frame_height:
            norm_padding = (frame_height - frame_width) / (2 * frame_height)
            peak_x = (peak_x - norm_padding) / (1.0 - (2 * norm_padding))
            peak_x = -0.5 / output_width if peak_x < 0.0 else peak_x
            peak_x = 1.0 if peak_x > 1.0 else peak_x

        frame_coords.append((body_parts[i], peak_x, peak_y))

    return frame_coords

def get_model(model_name):
    model_variant = model_name.lower()
    model_variant = model_variant[13:] if len(model_variant) > 7 else model_variant
    lite = True if model_variant.endswith('_lite') else False
    set_learning_phase(0)
    model = load_model(os.path.join('pose_estimation_models', 'efficientpose_models', 'keras', 'EfficientPose{0}.h5'.format(model_variant.upper())),
                       custom_objects={'BilinearWeights': keras_BilinearWeights, 'Swish': Swish(eswish),
                                       'eswish': eswish,'swish1': swish1})
    return model, {'rt': 224, 'i': 256, 'ii': 368, 'iii': 480, 'iv': 600, 'rt_lite': 224, 'i_lite': 256, 'ii_lite': 368}[model_variant], lite


def infer(batch, model, lite):
    if lite:
        batch_outputs = model.predict(batch)
    else:
        batch_outputs = model.predict(batch)[-1]
    return batch_outputs


def analyze(image, model, resolution, lite):
    image_height, image_width, _ = image.shape
    batch = np.expand_dims(image, axis=0)

    # Preprocess batch
    batch = preprocess(batch, resolution, lite)

    # Perform inference
    batch_outputs = infer(batch, model, lite)

    # Extract coordinates
    coordinates = [extract_coordinates(batch_outputs[0, ...], image_height, image_width)]
    return coordinates


def coordinates_to_landmarks(coordinates):
    landmarks = np.zeros((16, 2))
    for i in range(len(coordinates[0])):
        landmarks[i] = coordinates[0][i][1:]
    return landmarks
