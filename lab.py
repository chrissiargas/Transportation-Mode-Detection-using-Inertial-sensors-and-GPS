from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, ReLU, Add, MaxPooling1D
from tensorflow.keras.initializers import glorot_uniform

# Define the input shape
input_shape = (450, 10)

# Input tensor for the data
input_layer = Input(shape=input_shape)


# Function to create a convolutional block
def conv_block(input_tensor, kernel_size, filters, stage, block, s=2):
    # Naming convention for the layers
    conv_name_base = f'res{stage}{block}_branch'
    relu_name_base = f'relu{stage}{block}_branch'

    # Retrieve the number of filters for each convolutional layer
    F1, F2, F3 = filters

    # Save the input value for the shortcut
    x_shortcut = input_tensor

    # First component
    x = Conv1D(filters=F1, kernel_size=1, strides=s, padding='valid',
               name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(input_tensor)
    x = ReLU(name=relu_name_base + '2a')(x)

    # Second component
    x = Conv1D(filters=F2, kernel_size=kernel_size, strides=1, padding='same',
               name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(x)
    x = ReLU(name=relu_name_base + '2b')(x)

    # Third component
    x = Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid',
               name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(x)

    # Shortcut path
    x_shortcut = Conv1D(filters=F3, kernel_size=1, strides=s, padding='valid',
                        name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(x_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a ReLU activation
    x = Add()([x, x_shortcut])
    x = ReLU(name=relu_name_base + '2c')(x)

    return x


# Define the Residual blocks based on the provided architecture
# Block 1
x = conv_block(input_layer, kernel_size=3, filters=[64, 64, 256], stage=2, block='a', s=1)

# Max Pooling as per architecture
x = MaxPooling1D(pool_size=3, strides=2)(x)

# Block 2
x = conv_block(x, kernel_size=3, filters=[128, 128, 512], stage=3, block='a')

# Block 3
x = conv_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='a')

# Create model
model = Model(inputs=input_layer, outputs=x)

# Output the model summary
model.summary()
