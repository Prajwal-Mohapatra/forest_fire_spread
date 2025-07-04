from tensorflow.keras import layers, models, Input

def conv_block(x, filters, kernel_size=3, dilation=1):
    x = layers.Conv2D(filters, kernel_size, padding='same', dilation_rate=dilation)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding='same', dilation_rate=dilation)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def attention_gate(x, g, inter_channels):
    theta_x = layers.Conv2D(inter_channels, 1, strides=1, padding='same')(x)
    phi_g = layers.Conv2D(inter_channels, 1, strides=1, padding='same')(g)
    add = layers.Add()([theta_x, phi_g])
    relu = layers.ReLU()(add)
    psi = layers.Conv2D(1, 1, strides=1, padding='same', activation='sigmoid')(relu)
    return layers.Multiply()([x, psi])

def build_resunet_a(input_shape=(256, 256, 9)):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck (dilated)
    bn = conv_block(p4, 1024, dilation=2)

    # Decoder with attention
    u4 = layers.UpSampling2D((2, 2))(bn)
    a4 = attention_gate(c4, u4, 512)
    u4 = layers.Concatenate()([u4, a4])
    c5 = conv_block(u4, 512)

    u3 = layers.UpSampling2D((2, 2))(c5)
    a3 = attention_gate(c3, u3, 256)
    u3 = layers.Concatenate()([u3, a3])
    c6 = conv_block(u3, 256)

    u2 = layers.UpSampling2D((2, 2))(c6)
    a2 = attention_gate(c2, u2, 128)
    u2 = layers.Concatenate()([u2, a2])
    c7 = conv_block(u2, 128)

    u1 = layers.UpSampling2D((2, 2))(c7)
    a1 = attention_gate(c1, u1, 64)
    u1 = layers.Concatenate()([u1, a1])
    c8 = conv_block(u1, 64)

    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c8)

    model = models.Model(inputs, outputs)
    return model
