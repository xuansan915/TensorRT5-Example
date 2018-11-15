import uff
import sys
import argparse
import tensorrt as trt
import pycuda.driver as cuda # noinspection PyUnresolvedReferences
import pycuda.autoinit


# def main(argv):
    # '''
    # Optimize frozen TF graph and prepare a stand-alone TensorRT inference engine
    # '''
    # global parser

    # flags = parser(description="Parser to contain flags for running the TensorRT timers.").parse_args()
    # uff_model = uff.from_tensorflow_frozen_model(flags.frozen_graph, [flags.output_nodes])

    # with trt.Builder(trt.Logger(trt.Logger.INFO)) as builder:
        # builder.max_batch_size = flags.batch_size
        # builder.max_workspace_size = flags.workspace_size * (1 << 30)
        # builder.fp16_mode = True

        # with trt.UffParser() as parser:
            # parser.register_input(flags.input_nodes, flags.input_size)
            # parser.register_output(flags.output_nodes)
            # network = builder.create_network()
            # parser.parse_buffer(uff_model, network)
            # engine = builder.build_cuda_engine(network)

            # with open(flags.engine_path, 'wb') as f:
                # f.write(engine.serialize())

# class parser(argparse.ArgumentParser):

    # def __init__(self,description):
        # super(parser, self).__init__(description)

        # self.add_argument(
            # "--frozen_graph", "-fg", default="./model.pb",
            # help="[default: %(default)s] The location of a Frozen Graph ",
            # metavar="<FG>",
        # )

        # self.add_argument(
            # "--engine_path", "-od", default="./model.pb.plan",
            # help="[default: %(default)s] The location where output files will "
            # "be saved.",
            # metavar="<OD>",
        # )

        # self.add_argument(
            # "--input_nodes", "-in", default="input_1",
            # help="[default: %(default)s] The name of the graph input nodes , list"
            # "[input_image_shape, input_image]",
            # metavar="<IN>",
        # )

        # self.add_argument(
            # "--input_size", "-is", default=[28,28,1],
            # help="[default: %(default)s] The size of the graph input image",
            # metavar="<IS>",
        # )

        # self.add_argument(
            # "--output_nodes", "-on", default="dense_1/Softmax",
            # help="[default: %(default)s] The names of the graph output node, list "
            # "[input_image, boxes, scores, classes]",
            # metavar="<ON>",
        # )

        # self.add_argument(
            # "--fp16", action="store_true",
            # help="[default: %(default)s] If set, benchmark the model with TensorRT "
            # "using fp16 precision."
        # )

        # self.add_argument(
            # "--int8", action="store_true",
            # help="[default: %(default)s] If set, benchmark the model with TensorRT "
            # "using int8 precision."
        # )

        # self.add_argument(
            # "--workspace_size", "-ws", type=int, default=4,
            # help="[default: %(default)s] Workspace size in GB.",
            # metavar="<WS>"
        # )

        # self.add_argument(
            # "--batch_size", "-bs", type=int, default=1,
            # help="[default: %(default)s] Batch size for inference.",
            # metavar="<BS>"
        # )

# if __name__ == "__main__": main(argv=sys.argv)


### Settings
FROZEN_GDEF_PATH = './model/model.pb' # ADJUST
ENGINE_PATH = './model.pb.plan' # ADJUST
INPUT_NODE = 'input_1' # ADJUST
OUTPUT_NODE = 'dense_1/Softmax' # ADJUST
INPUT_SIZE = [28, 28, 1] # ADJUST
MAX_BATCH_SIZE = 1 # ADJUST
MAX_WORKSPACE = 4*(1 << 30) # ADJUST

### Convert TF frozen graph to UFF graph
uff_model = uff.from_tensorflow_frozen_model(FROZEN_GDEF_PATH, [OUTPUT_NODE])

### Create TRT model builder
trt_logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(trt_logger)
builder.max_batch_size = MAX_BATCH_SIZE
builder.max_workspace_size = MAX_WORKSPACE
builder.fp16_mode = True
# builder.int8_mode = True

### Create UFF parser
parser = trt.UffParser()
parser.register_input(INPUT_NODE, INPUT_SIZE)
parser.register_output(OUTPUT_NODE)

### Parse UFF graph
network = builder.create_network()
parser.parse_buffer(uff_model, network)

### Build optimized inference engine
engine = builder.build_cuda_engine(network)

### Save inference engine
with open(ENGINE_PATH, "wb") as f:
    f.write(engine.serialize())




