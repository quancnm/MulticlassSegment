from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model

def convolutional_block(input_layer, num_filters):
    """
    Apply a series of convolutional layers with batch normalization and ReLU activation.
    
    Returns:
        Tensor: Output tensor after passing through the convolutional block.
    """
    x = Conv2D(num_filters, 3, padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(input_layer, num_filters):
    """
    Create an encoder block consisting of a convolutional block followed by max pooling.
    
    Returns:
        Tuple: A tuple containing the output tensor of the convolutional block (skip connection)
               and the pooled tensor.
    """
    x = convolutional_block(input_layer, num_filters)
    pooled = MaxPool2D((2, 2))(x)
    return x, pooled

def decoder_block(input_layer, skip_features, num_filters):
    """
    Create a decoder block consisting of transpose convolution, concatenation with skip connection,
    and a convolutional block.
    Returns:
        Tensor: Output tensor after passing through the decoder block.
    """
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_layer)
    x = Concatenate()([x, skip_features])
    x = convolutional_block(x, num_filters)
    return x

def Unet(input_shape, num_classes):
    """
    Build a U-Net model.
    Returns:
        Model: U-Net model.
    """
    inputs = Input(input_shape)

    skip1, pool1 = encoder_block(inputs, 64)
    skip2, pool2 = encoder_block(pool1, 128)
    skip3, pool3 = encoder_block(pool2, 256)
    skip4, pool4 = encoder_block(pool3, 512)

    bottleneck = convolutional_block(pool4, 1024)

    up1 = decoder_block(bottleneck, skip4, 512)
    up2 = decoder_block(up1, skip3, 256)
    up3 = decoder_block(up2, skip2, 128)
    up4 = decoder_block(up3, skip1, 64)

    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(up4)

    model = Model(inputs, outputs, name="U-Net")
    return model

if __name__ == "__main__":
    input_shape = (512, 512, 3)
    num_classes = 20
    model = Unet(input_shape, num_classes)
    model.summary()
