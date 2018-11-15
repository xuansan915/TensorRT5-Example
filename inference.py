import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit # For automatic creation and cleanup of CUDA context
import tensorrt as trt
from model import process_dataset

ENGINE_PATH = './model/model.pb.plan' # ADJUST
INPUT_DATA_TYPE = np.float32 # ADJUST
TRT_LOGGER = trt.Logger()
x_train, y_train, x_test, y_test = process_dataset()
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

x_train, y_train, x_test, y_test = process_dataset()
print("real label: %d predict label: %d"  %(y_train[-1], np.argmax(infer(x_train[-1]))))

