from deepdreamtravel import DeepDreamTravel

if __name__ == '__main__':
    import argparse, sys

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('input_image', help='Input Image')
    parser.add_argument('output_video', help='Output location and name')
    # Optional arguments
    parser.add_argument('--node_switch', help='How often should layers switch', type=int)
    parser.add_argument('--resize_coeff', help='Zoom amount percentage', type=float)
    parser.add_argument('--show_iter', help='The image will be shown every X iterations', type=int)
    parser.add_argument('--offset', help='Left, Top, Right, Bottom zoom offset', nargs=4, type=int)
    parser.add_argument('--temp_dir', help='Temporary directory to store frames', type=int)
    parser.add_argument('--start_iter', help='If generation was interrupted, enter the next iteration here', type=int)
    parser.add_argument('--start_index', help='Which layer index should be used first. Also used when interruption occured', type=int)
    parser.add_argument('--start_offset', help='In-case there are invalid layers, enter the offset given by terminal output when interruption happened', type=int)
    parser.add_argument('--max_iteration', help='If false, all layers will be "explored". Else set maximum number of iterations', type=int)
    parser.add_argument('--octaves', help='Higher number leads to bigger ("better") visuals. Takes significantly more time to generate! Lower number in-case of errors with small images', type=int)
    parser.add_argument('--iter_n', help='Number of iterations per octave', type=int)
    parser.add_argument('--octave_scale', help='Resize amount per octave. Higher number leads to higher dream states', type=int)
    parser.add_argument('--fps', help='Frames per second of output video', type=int)
    parser.add_argument('--delete_temp', help='Should generated images be cleaned up after video is created?', type=bool)
    parser.add_argument('--max_memory', help='Maximum amount of memory to use in GB', type=int)

    parser.add_argument('--protxt', help='Protxt file. Defaults to deploy_places205.protxt', default='deploy_places205.protxt')
    parser.add_argument('--caffemodel', help='Caffe Model. Defaults to googlelet_places205_train_iter_2400000.caffemodel', default='googlelet_places205_train_iter_2400000.caffemodel')

    args = parser.parse_args()

    # Collecting all arguments for object initialization
    model_dict = [args.protxt, args.caffemodel]

    # Collecting all specified arguments for generate function
    generate_dict = {k: v for k, v in vars(args).items() if v is not None and k not in ['protxt', 'caffemodel']}

    dreamer = DeepDreamTravel(*model_dict)
    dreamer.generate(**generate_dict)