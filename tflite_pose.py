#
# skeleton code from google tensorflow lite inference tutorial
import os
import numpy as np
import tensorflow as tf
from PIL import Image
'''
def ParseOutput(output):
    inference_time, output = output
    #outputs = [output[i:j] for i, j in zip(self._output_offsets, self._output_offsets[1:])]
    outputs=output    
    keypoints = outputs[0].reshape(-1, len(KEYPOINTS), 2)
    keypoint_scores = outputs[1].reshape(-1, len(KEYPOINTS))
    pose_scores = outputs[2]
    nposes = int(outputs[3][0])
    assert nposes < outputs[0].shape[0]

    # Convert the poses to a friendlier format of keypoints with associated
    # scores.
    poses = []
    for pose_i in range(nposes):
        keypoint_dict = {}
        for point_i, point in enumerate(keypoints[pose_i]):
            keypoint = Keypoint(KEYPOINTS[point_i], point,
                                keypoint_scores[pose_i, point_i])
            if self._mirror: keypoint.yx[1] = self.image_width - keypoint.yx[1]
            keypoint_dict[KEYPOINTS[point_i]] = keypoint
        poses.append(Pose(keypoint_dict, pose_scores[pose_i]))

    return poses, inference_time
    '''


''' Main '''

# Load the tflite model and allocate tensors.
model_path_arg = 'models/posenet_mobilenet_v1_075_481_641_quant.tflite'

interpreter = tf.lite.Interpreter(model_path=model_path_arg)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details= interpreter.get_output_details()

# Test the model on test input image (TODO: inputs as an argument)
pil_image = Image.open('couple.jpg')
pil_image = pil_image.resize((641, 481), Image.NEAREST)

input_shape = input_details[0]['shape']

input_data = np.uint8(pil_image)
input_data = input_data.reshape([1, w, h, d])
'''
# Extend or crop the input to match the input shape of the network.
if img.shape[0] < self.image_height or img.shape[1] < self.image_width:
    img = np.pad(img, [[0, max(0, self.image_height - img.shape[0])],
                       [0, max(0, self.image_width - img.shape[1])], [0, 0]],
                 mode='constant')
img = img[0:self.image_height, 0:self.image_width]
assert (img.shape == tuple(self._input_tensor_shape[1:]))
'''
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
# The function 'get_tensor()' returns a copy of the tensor data.
# Use 'tensor()' in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data[0])
print(len(output_data[0]))
#print(ParseOutput(
