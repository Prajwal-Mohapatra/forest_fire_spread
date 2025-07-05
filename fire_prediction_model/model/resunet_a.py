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

def atrous_spatial_pyramid_pooling(x, output_filters=256, rates=[6, 12, 18], use_global_pooling=True):
    """
    Enhanced ASPP module with configurable dilation rates and optional global pooling.
    
    Args:
        x: Input tensor
        output_filters: Number of output filters for each branch
        rates: List of dilation rates for parallel convolutions
        use_global_pooling: Whether to include global average pooling branch
    """
    branches = []
    
    # 1x1 convolution branch
    conv_1x1 = layers.Conv2D(output_filters, 1, use_bias=False)(x)
    conv_1x1 = layers.BatchNormalization()(conv_1x1)
    conv_1x1 = layers.ReLU()(conv_1x1)
    branches.append(conv_1x1)
    
    # Multi-scale dilated convolutions
    for rate in rates:
        conv = layers.Conv2D(output_filters, 3, padding='same', 
                           dilation_rate=rate, use_bias=False)(x)
        conv = layers.BatchNormalization()(conv)
        conv = layers.ReLU()(conv)
        branches.append(conv)
    
    # Global average pooling branch (optional)
    if use_global_pooling:
        # Create a custom layer for global pooling and resizing
        class GlobalPoolingBranch(layers.Layer):
            def __init__(self, filters, **kwargs):
                super().__init__(**kwargs)
                self.filters = filters
                self.global_pool = layers.GlobalAveragePooling2D(keepdims=True)
                self.conv = layers.Conv2D(filters, 1, use_bias=False)
                self.bn = layers.BatchNormalization()
                self.relu = layers.ReLU()
            
            def call(self, inputs):
                # Get input shape
                input_shape = tf.shape(inputs)
                height, width = input_shape[1], input_shape[2]
                
                # Global pooling
                pooled = self.global_pool(inputs)
                pooled = self.conv(pooled)
                pooled = self.bn(pooled)
                pooled = self.relu(pooled)
                
                # Resize using tf.image.resize wrapped in Lambda layer
                def resize_fn(x):
                    return tf.image.resize(x, [height, width], method='bilinear')
                
                resized = layers.Lambda(resize_fn)(pooled)
                return resized
        
        global_branch = GlobalPoolingBranch(output_filters)(x)
        branches.append(global_branch)
    
    # Concatenate all branches
    concat = layers.Concatenate()(branches)
    
    # Final projection to reduce channel dimension
    output = layers.Conv2D(output_filters, 1, use_bias=False)(concat)
    output = layers.BatchNormalization()(output)
    output = layers.ReLU()(output)
    
    # Optional dropout for regularization
    output = layers.Dropout(0.1)(output)
    
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

def build_resunet_a(input_shape=(256, 256, 9), num_classes=1, use_enhanced_aspp=False):
    """
    ResUNet-A implementation with proper residual blocks, ASPP, and attention gates.
    
    Args:
        input_shape: Input tensor shape
        num_classes: Number of output classes
        use_enhanced_aspp: Whether to use the enhanced ASPP module in the bridge
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

    # Bridge with optional enhanced ASPP
    if use_enhanced_aspp:
        # Use enhanced ASPP module
        bridge = atrous_spatial_pyramid_pooling(p4, output_filters=1024, 
                                              rates=[6, 12, 18], 
                                              use_global_pooling=True)
    else:
        # Original multi-scale dilated convolutions
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
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid' if num_classes == 1 else 'softmax')(c8)

    model = models.Model(inputs, outputs, name='ResUNet_A')
    return model
