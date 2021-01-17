# Deep Dream Travel

Deep Dream Travel allows you to turn a single image into a video, diving through various layers of a neural network.

This effect is achieved by recursively feeding the image into the network and regularly switching the targeted layer.

## Demo

![Deep Dream Demo](https://drive.google.com/uc?export=view&id=1sHDkmz13IQlp-M2KxSlwI6htXaL04wnH)

### Higher resolution video
A higher resolution example can be found on [YouTube](https://www.youtube.com/watch?v=VjcBpVmRm9Y).

## Installation
Download the code and look at `example.py`.

Linux instalation may require some aditional steps. Please check out this [Tutorial](https://www.youtube.com/watch?v=f1HLevIo0Z8) by [Nerdy Rodent](https://www.youtube.com/channel/UC4-5v-f-xKnbi1yaAuRSi_w).

Or ask for [support or post questions/requests here](https://hackcommunity.net/t/trippy-videos-with-deep-dream-travel)
## Dependencies

* [PIL](https://pillow.readthedocs.io/en/stable/)
* [SciPy](https://pypi.org/project/scipy/)
* [Caffe](https://caffe.berkeleyvision.org/)
* [OpenCV](https://pypi.org/project/opencv-python/)

## Usage

```python
from deepdreamtravel import DeepDreamTravel

dreamer = DeepDreamTravel('protxt', 'caffemodel') # View example.py for an example
dreamer.generate(input_image="noise.jpg",
                 node_switch=10, # How often should layers switch
                 resize_coeff=0.05, # Zoom amount in %
                 show_iter=100, # The image will be shown every X iterations
                 offset=[0,0,0,0], # Left, Top, Right, Bottom zoom offset
                 temp_dir="tmp", # Temp directory to store frames
                 start_iter=0, # If generation was interrupted, enter the next iteration here
                 start_index=0, # Which layer index should be used first. Also used when interruption occured
                 start_offset=0, # In-case there are invalid layers, enter the offset given by terminal output when interruption happened.
                 max_iteration=False, # If false, all layers will be "explored". Else set maximum number of iterations
                 octaves=4, # Higher number leads to bigger ("better") visuals. Takes significantly more time to generate! Lower number in-case of errors with small images.
                 iter_n=10, # Number of iterations per octave.
                 octave_scale=1.4, # Resize amount per octave. Higher number leads to higher dream states.
                 output_video="DeepDreamTravel.avi", # Output location and name. Don't forget to end with .avi!
                 fps=30, # Frames per second
                 delete_temp=True, # Clean up generated images after video was generated?
                 max_memory=3 # Maximum amount of memory to use in GB
)
```
**Note:** The examples were generated with [Places205-GoogLeNet](http://places.csail.mit.edu/downloadCNN.html). You can find other interesting models [here](https://github.com/BVLC/caffe/wiki/Model-Zoo).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
