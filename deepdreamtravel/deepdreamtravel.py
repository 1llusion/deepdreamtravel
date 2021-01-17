from io import BytesIO
import PIL.Image
import io
import os
import psutil
from google.protobuf import text_format
import scipy.ndimage as nd
import caffe
from .utility import *
from deepdreamtravel.imagetovideo import ImageToVideo
from random import randint
from pathlib import Path

class DeepDreamTravel:
    objectives = []

    def __init__(self, prototxt, caffemodel, model_path="./"):
        caffe.set_device(0)
        caffe.set_mode_gpu()

        net_fn = model_path + prototxt
        param_fn = model_path + caffemodel

        # Patching model to be able to compute gradients.
        # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
        model = caffe.io.caffe_pb2.NetParameter()
        text_format.Merge(open(net_fn).read(), model)
        model.force_backward = True
        open('tmp.prototxt', 'w').write(str(model))

        self.net = caffe.Classifier('tmp.prototxt', param_fn,
                               mean=np.float32([104.0, 116.0, 122.0]),  # ImageNet mean, training set dependent
                               channel_swap=(2, 1, 0))  # the reference model has channels in BGR order instead of RGB

        self.nodes = list(self.net._layer_names)
        self.nodes = self.nodes[1:-2]
        self.node_len = len(self.nodes)
        self.objectives.append(objective_L2)
        print("Loaded ", self.node_len, "nodes")

    def register_objective(self, func):
        """
        Register a new objective to be used by deep dream
        :function func:
        :return: True if successful else False
        """
        if not hasattr(func, '__call__'):
            return False

        self.objectives.append(func)
        return True

    def make_step(self, net, step_size=1.5, end='inception_4c/output',
                  jitter=32, clip=True, objective=objective_L2):
        '''Basic gradient ascent step.'''

        src = net.blobs['data']  # input image is stored in Net's 'data' blob
        dst = net.blobs[end]

        ox, oy = np.random.randint(-jitter, jitter + 1, 2)
        src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)  # apply jitter shift

        net.forward(end=end)
        objective(dst)  # specify the optimization objective
        net.backward(start=end)
        g = src.diff[0]
        # apply normalized ascent step to the input image
        src.data[:] += step_size / np.abs(g).mean() * g

        src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)  # unshift image

        if clip:
            bias = net.transformer.mean['data']
            src.data[:] = np.clip(src.data, -bias, 255 - bias)

    def list_nodes(self, display=True):
        """
        :bool display: Print out all nodes
        :return: list of nodes
        """
        if display:
            print("The following nodes are loaded:", self.nodes)
        return self.nodes

    def showarray(self, a, fmt='jpeg'):
        # clip the values to be between 0 and 255
        a = np.uint8(np.clip(a, 0, 255))
        f = BytesIO()
        PIL.Image.fromarray(a).save(f, fmt)
        return f.getvalue()

    def deepdream(self, net, base_img, iter_n=10, octave_n=4, octave_scale=1.4,
                     end='inception_4c/output', clip=True, **step_params):
        # prepare base images for all octaves
        octaves = [preprocess(net, base_img)]
        for i in range(octave_n - 1):
            octaves.append(nd.zoom(octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

        src = net.blobs['data']
        detail = np.zeros_like(octaves[-1])  # allocate image for network-produced details
        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]
            if octave > 0:
                # upscale details from the previous octave
                h1, w1 = detail.shape[-2:]
                detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=1)

            src.reshape(1, 3, h, w)  # resize the network's input image size
            src.data[0] = octave_base + detail
            for i in range(iter_n):
                self.make_step(net, end=end, clip=clip, **step_params)

                # visualization
                vis = deprocess(net, src.data[0])
                if not clip:  # adjust image contrast if clipping is disabled
                    vis = vis * (255.0 / np.percentile(vis, 99.98))
                ret = self.showarray(vis)

            # extract details produced on the current octave
            detail = src.data[0] - octave_base
        # returning the resulting image
        return ret

    def generate(self,
                 input_image="noise.jpg",
                 node_switch=10,
                 resize_coeff=0.05,
                 show_iter=100,
                 offset=[0,0,0,0],
                 temp_dir="tmp",
                 start_iter=0,
                 start_index=0,
                 start_offset=0,
                 max_iteration=False,
                 octaves=4,
                 iter_n=10,
                 octave_scale=1.4,
                 output_video="DeepDreamTravel.avi",
                 fps=30,
                 delete_temp=True,
                 max_memory=3
                 ):
        # Loading the image
        img0 = PIL.Image.open(input_image)

        w, h = img0.size
        img0 = np.float32(img0)

        images_arr = [img0]  # Storing all images in an array

        node_switch = node_switch  # How often should nodes be switched

        image_resize_coeff = resize_coeff
        max_memory = max_memory  # in GB
        show_image = show_iter  # Show image every X iterations
        offset = offset  # Left, Top, Right, Bottom

        directory = Path(temp_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Gettin process ID to monitor memory
        process = psutil.Process(os.getpid())

        iteration = start_iter
        index = start_index  # Starting at first node (start at 41)
        index_offset = start_offset  # To display true index from the original list add offset

        # If Max iteration is not set, the video will include all layers
        max_iter = max_iteration if max_iteration is not False else self.node_len * node_switch
        if max_iteration is False:
            print("Max iteration not set. Setting to", max_iter)

        objective_index_max = len(self.objectives) # Number of registered objectives
        while iteration < max_iter:
            node = self.nodes[index]
            # Getting a randomg objective
            objective = self.objectives[randint(0, objective_index_max - 1)]

            if (process.memory_info().rss * 10 ** -9) >= max_memory:
                images_arr = save_images(images_arr, iteration, output_dir=directory)

            try:
                for octave in range(1, octaves + 1):
                    for _ in range(int(node_switch / octaves)):
                        print(node, "iteration", iteration, "index", index, "offset", index_offset)
                        img0 = self.deepdream(self.net, images_arr[-1], iter_n=iter_n, octave_n=octave,
                                              octave_scale=octave_scale, end=node, objective=objective)
                        img0 = PIL.Image.open(io.BytesIO(img0))

                        if not iteration % show_image:
                            img0.show()
                        # Cropping image
                        img0 = nd.affine_transform(img0, [1 - image_resize_coeff + offset[0], 1 - image_resize_coeff + offset[1], 1],
                                                   [h * image_resize_coeff / 2 - offset[2], w * image_resize_coeff / 2 - offset[3], 0], order=1)
                        # Resizing image
                        img1 = np.float32(img0)
                        images_arr.append(img1)
                        iteration += 1
            except (ValueError, KeyError):
                print("Invalid node")
                index_offset += 1
                del self.nodes[index]
                self.node_len = len(self.nodes)

                # Adjusting max iteration
                if max_iteration is False:
                    max_iter = self.node_len * node_switch
                    print("Adjusted max iteration to", max_iter)
                continue
            index += 1
            if index >= self.node_len:
                print("Too large node.")
                index = 0
        save_images(images_arr, iteration, output_dir=directory)

        # Generating video
        print("Generating video...")
        # Making sure the output_video ends with .avi
        if Path(output_video).suffix != '.avi':
            output_video += '.avi'

        file_generated = ImageToVideo.convert(temp_dir, output_video, fps=fps, delete_files=delete_temp)
        if file_generated:
            print("Video generated to", output_video)
        else:
            print("Error File not generated.")