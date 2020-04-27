from deepdreamtravel import DeepDreamTravel
dreamer = DeepDreamTravel('deploy_places205.protxt', 'googlelet_places205_train_iter_2400000.caffemodel')
dreamer.generate(delete_temp=True)