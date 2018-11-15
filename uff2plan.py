import tensorrt as trt

### Settings
UFF_PATH = 'models/model.uff' # ADJUST
ENGINE_PATH = 'models/model.uff.plan' # ADJUST
INPUT_NODE = 'input_1' # ADJUST
OUTPUT_NODE = 'dense_1/Softmax' # ADJUST
INPUT_SIZE = [28, 28, 1] # ADJUST
MAX_BATCH_SIZE = 1 # ADJUST
MAX_WORKSPACE = 4*(1 << 30) # ADJUST

### Create TRT model builder
trt_logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(trt_logger)
builder.max_batch_size = MAX_BATCH_SIZE
builder.max_workspace_size = MAX_WORKSPACE
builder.fp16_mode = True

### Create UFF parser
parser = trt.UffParser()
parser.register_input(INPUT_NODE, INPUT_SIZE)
parser.register_output(OUTPUT_NODE)

### Parse UFF graph
network = builder.create_network()
parser.parse(UFF_PATH, network)

### Build optimized inference engine
engine = builder.build_cuda_engine(network)

### Save inference engine
with open(ENGINE_PATH, "wb") as f:
    f.write(engine.serialize())



