# Keypoints with binary descriptors for one RGB-D frame (feature-level input). Used by the simulator's
# bags and by external feature extractors; without it the node extracts ORB from the RGB image.
std_msgs/Header header
float32[] u                # pixel coordinates
float32[] v
float32[] response
uint8[] descriptors        # N x 32 bytes (256-bit binary descriptors), row-major
