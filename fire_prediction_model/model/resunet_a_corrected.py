from tensorflow.keras import layers, models, Input
import tensorflow as tf

def residual_block(x, filters, kernel_size=3, dilation=1, stride=1):
    """
    True residual block with skip connections as per ResNet architecture.
    """
    shortcut = x
    
    # Main path
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', 
                     dilation_rate=dilation, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same', 
                     dilation_rate=dilation, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    # Adjust shortcut dimensions if needed
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Residual connection
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def atrous_spatial_pyramid_pooling(x, output_filters=256):
    """
    ASPP module for multi-scale feature extraction.
    """
    # Image-level features
    image_pooling = layers.GlobalAveragePooling2D(keepdims=True)(x)
    image_pooling = layers.Conv2D(output_filters, 1, use_bias=False)(image_pooling)
    image_pooling = layers.BatchNormalization()(image_pooling)
    image_pooling = layers.ReLU()(image_pooling)
    image_pooling = layers.UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation='bilinear')(image_pooling)
    
    # Multi-scale dilated convolutions
    conv_1x1 = layers.Conv2D(output_filters, 1, use_bias=False)(x)
    conv_1x1 = layers.BatchNormalization()(conv_1x1)
    conv_1x1 = layers.ReLU()(conv_1x1)
    
    conv_3x3_1 = layers.Conv2D(output_filters, 3, padding='same', dilation_rate=6, use_bias=False)(x)
    conv_3x3_1 = layers.BatchNormalization()(conv_3x3_1)
    conv_3x3_1 = layers.ReLU()(conv_3x3_1)
    
    conv_3x3_2 = layers.Conv2D(output_filters, 3, padding='same', dilation_rate=12, use_bias=False)(x)
    conv_3x3_2 = layers.BatchNormalization()(conv_3x3_2)
    conv_3x3_2 = layers.ReLU()(conv_3x3_2)
    
    conv_3x3_3 = layers.Conv2D(output_filters, 3, padding='same', dilation_rate=18, use_bias=False)(x)
    conv_3x3_3 = layers.BatchNormalization()(conv_3x3_3)
    conv_3x3_3 = layers.ReLU()(conv_3x3_3)
    
    # Concatenate all features
    concat = layers.Concatenate()([image_pooling, conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3])
    
    # Final projection
    output = layers.Conv2D(output_filters, 1, use_bias=False)(concat)
    output = layers.BatchNormalization()(output)
    output = layers.ReLU()(output)
    
    return output

def improved_attention_gate(x, g, inter_channels):
    """
    Improved attention gate with proper gating mechanism.
    """
    # Ensure spatial dimensions match
    if x.shape[1] != g.shape[1] or x.shape[2] != g.shape[2]:
        g = layers.UpSampling2D(size=(x.shape[1]//g.shape[1], x.shape[2]//g.shape[2]))(g)
    
    # Feature projection
    theta_x = layers.Conv2D(inter_channels, 1, strides=1, padding='same', use_bias=False)(x)
    phi_g = layers.Conv2D(inter_channels, 1, strides=1, padding='same', use_bias=False)(g)
    
    # Attention computation
    add_xg = layers.Add()([theta_x, phi_g])
    relu_xg = layers.ReLU()(add_xg)
    
    # Attention coefficients
    psi = layers.Conv2D(1, 1, strides=1, padding='same', use_bias=False)(relu_xg)
    psi = layers.BatchNormalization()(psi)
    attention_coeffs = layers.Activation('sigmoid')(psi)
    
    # Apply attention
    attended_x = layers.Multiply()([x, attention_coeffs])
    
    return attended_x

def build_resunet_a_corrected(input_shape=(256, 256, 9), num_classes=1):
    """
    Corrected ResUNet-A implementation with proper residual blocks and ASPP.
    """
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Encoder with residual blocks
    # Stage 1
    e1 = residual_block(x, 64)
    e1 = residual_block(e1, 64)
    
    # Stage 2
    e2 = residual_block(e1, 128, stride=2)
    e2 = residual_block(e2, 128)
    
    # Stage 3
    e3 = residual_block(e2, 256, stride=2)
    e3 = residual_block(e3, 256)
    
    # Stage 4
    e4 = residual_block(e3, 512, stride=2)
    e4 = residual_block(e4, 512)
    
    # Bridge with ASPP
    bridge = atrous_spatial_pyramid_pooling(e4, output_filters=512)
    
    # Decoder with attention gates
    # Stage 4 decode
    d4 = layers.UpSampling2D(2, interpolation='bilinear')(bridge)
    a4 = improved_attention_gate(e4, d4, 256)
    d4 = layers.Concatenate()([d4, a4])
    d4 = residual_block(d4, 512)
    d4 = residual_block(d4, 512)
    
    # Stage 3 decode
    d3 = layers.UpSampling2D(2, interpolation='bilinear')(d4)
    a3 = improved_attention_gate(e3, d3, 128)
    d3 = layers.Concatenate()([d3, a3])
    d3 = residual_block(d3, 256)
    d3 = residual_block(d3, 256)
    
    # Stage 2 decode
    d2 = layers.UpSampling2D(2, interpolation='bilinear')(d3)
    a2 = improved_attention_gate(e2, d2, 64)
    d2 = layers.Concatenate()([d2, a2])
    d2 = residual_block(d2, 128)
    d2 = residual_block(d2, 128)
    
    # Stage 1 decode
    d1 = layers.UpSampling2D(2, interpolation='bilinear')(d2)
    a1 = improved_attention_gate(e1, d1, 32)
    d1 = layers.Concatenate()([d1, a1])
    d1 = residual_block(d1, 64)
    d1 = residual_block(d1, 64)
    
    # Final upsampling to original resolution
    final = layers.UpSampling2D(4, interpolation='bilinear')(d1)
    
    # Output head
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid' if num_classes == 1 else 'softmax')(final)
    
    model = models.Model(inputs, outputs, name='ResUNet_A_Corrected')
    return model

# Alternative simplified but correct implementation
def build_resunet_a_simplified(input_shape=(256, 256, 9)):
    """
    Simplified but architecturally correct ResUNet-A.
    """
    inputs = Input(shape=input_shape)

    # Encoder with residual blocks
    c1 = residual_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = residual_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = residual_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = residual_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bridge with multi-scale dilated convolutions
    bridge1 = residual_block(p4, 1024, dilation=1)
    bridge2 = residual_block(p4, 1024, dilation=2)
    bridge3 = residual_block(p4, 1024, dilation=4)
    bridge = layers.Add()([bridge1, bridge2, bridge3])

    # Decoder with attention
    u4 = layers.UpSampling2D((2, 2))(bridge)
    a4 = improved_attention_gate(c4, u4, 512)
    u4 = layers.Concatenate()([u4, a4])
    c5 = residual_block(u4, 512)

    u3 = layers.UpSampling2D((2, 2))(c5)
    a3 = improved_attention_gate(c3, u3, 256)
    u3 = layers.Concatenate()([u3, a3])
    c6 = residual_block(u3, 256)

    u2 = layers.UpSampling2D((2, 2))(c6)
    a2 = improved_attention_gate(c2, u2, 128)
    u2 = layers.Concatenate()([u2, a2])
    c7 = residual_block(u2, 128)

    u1 = layers.UpSampling2D((2, 2))(c7)
    a1 = improved_attention_gate(c1, u1, 64)
    u1 = layers.Concatenate()([u1, a1])
    c8 = residual_block(u1, 64)

    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c8)

    model = models.Model(inputs, outputs, name='ResUNet_A_Simplified')
    return model
