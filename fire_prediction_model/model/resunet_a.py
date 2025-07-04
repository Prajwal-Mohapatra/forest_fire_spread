from keras_unet_collection import models

def build_resunet_a(input_shape=(256, 256, 9)):
    model = models.resunet_a_2d(
        input_shape,
        filter_num=[32, 64, 128, 256],
        dilation_num=[1, 2, 4, 8],
        n_labels=1,
        stack_num_down=2,
        stack_num_up=2,
        activation='ReLU',
        output_activation='Sigmoid',
        batch_norm=True,
        pool=True,
        unpool=True
    )
    return model
