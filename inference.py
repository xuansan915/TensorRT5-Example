import time
import numpy as np
import tensorflow as tf
import pycuda.driver as cuda
import pycuda.autoinit # For automatic creation and cleanup of CUDA context
import tensorrt as trt

ENGINE_PATH = './models/model.pb.plan' # ADJUST
ENGINE_PATH = './models/model.uff.plan' # ADJUST
INPUT_DATA_TYPE = np.float32 # ADJUST
TRT_LOGGER = trt.Logger()
engine_buff = open(ENGINE_PATH, 'rb').read()

### Prepare TRT execution context, CUDA stream and necessary buffers
with trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(engine_buff)

context = engine.create_execution_context()
stream = cuda.Stream()
host_in = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=INPUT_DATA_TYPE)
host_out = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=INPUT_DATA_TYPE)
devide_in = cuda.mem_alloc(host_in.nbytes)
devide_out = cuda.mem_alloc(host_out.nbytes)

### Run inference
def infer(img):
    bindings = [int(devide_in), int(devide_out)]
    np.copyto(host_in, img.ravel())
    cuda.memcpy_htod_async(devide_in, host_in, stream)
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_out, devide_out, stream)
    stream.synchronize()
    return host_out

images, labels = tf.keras.datasets.mnist.load_data()[1]
images = np.reshape(images, [-1, 28, 28, 1]) / 255.0

print("real label: %d predict label: %d"  %(labels[-1], np.argmax(infer(images[-1]))))

