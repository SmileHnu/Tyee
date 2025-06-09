import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper Activation Function ---
def get_activation(name: str) -> nn.Module:
    """Returns a PyTorch activation module based on its name."""
    if name.lower() == 'relu':
        return nn.ReLU()
    elif name.lower() == 'tanh':
        return nn.Tanh()
    elif name.lower() == 'sigmoid':
        return nn.Sigmoid()
    # Softmax is usually applied with dimension specified, handle separately if needed
    else:
        raise ValueError(f"Unsupported activation: {name}")

# --- Basic Convolution Block ---
class ConvBNActivation(nn.Module):
    """Convolution -> Batch Normalization -> Activation Block"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size, padding,
                 dilation_rate: int = 1, activation_name: str = 'relu', is_1d_kernel_on_height=True):
        super().__init__()
        
        actual_kernel_size = (kernel_size, 1) if is_1d_kernel_on_height else kernel_size
        actual_dilation = (dilation_rate, dilation_rate) if is_1d_kernel_on_height and isinstance(dilation_rate, int) else dilation_rate
        if isinstance(actual_dilation, int): # Keras code uses (dilation_rate, dilation_rate) for 1D kernel too
             actual_dilation = (dilation_rate, dilation_rate)


        # Calculate padding for 'same' effect
        if padding == 'same':
            # For stride 1, padding = (dilation * (kernel_size - 1)) / 2
            pad_h = (actual_dilation[0] * (actual_kernel_size[0] - 1)) // 2
            pad_w = (actual_dilation[1] * (actual_kernel_size[1] - 1)) // 2
            actual_padding = (pad_h, pad_w)
        elif isinstance(padding, int):
            actual_padding = padding
        else: # Assume padding is already a tuple (pad_h, pad_w) or specific value like 0
            actual_padding = padding

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=actual_kernel_size,
                              padding=actual_padding, dilation=actual_dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation_name)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

# --- Upsampling Block ---
class UpsampleBlock(nn.Module):
    """Upsamples the input tensor to the size of the target_skip_tensor."""
    def __init__(self, mode='bilinear', align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x, target_skip_tensor):
        # target_skip_tensor provides the target H, W for upsampling
        # PyTorch NCHW format: target_skip_tensor.size(2) is H, target_skip_tensor.size(3) is W
        return F.interpolate(x, size=(target_skip_tensor.size(2), target_skip_tensor.size(3)),
                             mode=self.mode, align_corners=self.align_corners)

# --- U-Encoder Block (U-Unit) ---
class UEncoderBlock(nn.Module):
    """U-unit: a repetitive sub-structure of SalientSleepNet (PyTorch version)."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pooling_size: int,
                 middle_layer_channels: int, depth: int, padding: str = 'same', activation_name: str = 'relu'):
        super().__init__()
        self.depth = depth
        self.activation_name = activation_name
        self.padding = padding
        self.kernel_size = kernel_size
        
        # Initial convolution (conv_bn0 in Keras)
        self.conv_bn0 = ConvBNActivation(in_channels, out_channels, kernel_size, padding, activation_name=activation_name)

        # Encoder path
        self.encoder_convs = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        current_channels = out_channels
        for d in range(depth - 1):
            conv = ConvBNActivation(current_channels, middle_layer_channels, kernel_size, padding, activation_name=activation_name)
            self.encoder_convs.append(conv)
            current_channels = middle_layer_channels
            if d != depth - 2: # No pooling after the second to last encoder conv
                pool = nn.MaxPool2d(kernel_size=(pooling_size, 1)) # Keras code uses (pooling_size, 1)
                self.encoder_pools.append(pool)

        # Bottleneck convolution
        self.bottleneck_conv = ConvBNActivation(current_channels, middle_layer_channels, kernel_size, padding, activation_name=activation_name)

        # Decoder path
        self.decoder_upsamples = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        current_channels = middle_layer_channels
        for d in range(depth - 1, 0, -1):
            self.decoder_upsamples.append(UpsampleBlock())
            # Determine channels for concat: upsampled channels + skip connection channels
            # Skip connection comes from encoder_convs output (middle_layer_channels)
            concat_channels = current_channels + middle_layer_channels
            
            # Determine out_channels for this decoder conv
            # If d == 1, this is the last decoder conv before residual add, its out_channels should match conv_bn0's out_channels
            decoder_out_channels = out_channels if d == 1 else middle_layer_channels
            
            conv = ConvBNActivation(concat_channels, decoder_out_channels, kernel_size, padding, activation_name=activation_name)
            self.decoder_convs.append(conv)
            current_channels = decoder_out_channels
            
        self.upsample_final = UpsampleBlock() # For the final upsample before residual add if needed (not explicitly in Keras U-unit logic but implied by structure)


    def forward(self, x):
        # Initial conv
        x0 = self.conv_bn0(x)

        # Encoder path
        skip_connections = []
        encoded_x = x0
        for i in range(self.depth - 1):
            encoded_x = self.encoder_convs[i](encoded_x)
            skip_connections.append(encoded_x)
            if i != self.depth - 2:
                encoded_x = self.encoder_pools[i](encoded_x)
        
        # Bottleneck
        bottleneck_out = self.bottleneck_conv(encoded_x)

        # Decoder path
        decoded_x = bottleneck_out
        for i in range(self.depth - 1):
            skip_connection = skip_connections.pop() # Pop in reverse order of appending
            # The upsample target is the skip connection
            decoded_x = self.decoder_upsamples[i](decoded_x, skip_connection)
            decoded_x = torch.cat([decoded_x, skip_connection], dim=1) # dim=1 for channels
            decoded_x = self.decoder_convs[i](decoded_x)
            
        # Residual connection
        return decoded_x + x0


# --- Multi-Scale Extraction (MSE) Block ---
class MSEBlock(nn.Module):
    """Multi-scale Extraction Module (PyTorch version)."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation_rates: list, padding: str = 'same', activation_name: str = "relu"):
        super().__init__()
        self.convs = nn.ModuleList()
        for dr in dilation_rates:
            self.convs.append(
                ConvBNActivation(in_channels, out_channels, kernel_size, padding,
                                 dilation_rate=dr, activation_name=activation_name)
            )
        
        concatenated_channels = out_channels * len(dilation_rates)
        self.down_conv1 = nn.Conv2d(concatenated_channels, out_channels * 2,
                                    kernel_size=(kernel_size, 1), padding=self._calculate_same_padding(kernel_size, 1))
        self.act1 = get_activation(activation_name)
        self.down_conv2 = nn.Conv2d(out_channels * 2, out_channels,
                                    kernel_size=(kernel_size, 1), padding=self._calculate_same_padding(kernel_size, 1))
        self.act2 = get_activation(activation_name)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def _calculate_same_padding(self, kernel_h, kernel_w, dilation_h=1, dilation_w=1):
        pad_h = (dilation_h * (kernel_h - 1)) // 2
        pad_w = (dilation_w * (kernel_w - 1)) // 2
        return (pad_h, pad_w)

    def forward(self, x):
        dilated_conv_outputs = [conv(x) for conv in self.convs]
        concatenated = torch.cat(dilated_conv_outputs, dim=1) # Concatenate along channel dim

        down = self.down_conv1(concatenated)
        down = self.act1(down)
        down = self.down_conv2(down)
        down = self.act2(down)
        out = self.bn_out(down)
        return out

# --- Single Branch of SalientSleepNet ---
class Branch(nn.Module):
    """Builds one branch of the SalientSleepNet (PyTorch version)."""
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.padding = config['train']['padding']
        self.activation_name = config['train']['activation']
        self.kernel_size = config['train']['kernel_size']
        self.filters = config['train']['filters'] # e.g. [16, 32, 64, 128, 256]
        self.pooling_sizes = config['train']['pooling_sizes'] # e.g. [10, 8, 6, 4]
        self.u_depths = config['train']['u_depths'] # e.g. [4,4,4,4]
        self.u_inner_filter = config['train']['u_inner_filter']
        self.mse_filters = config['train']['mse_filters'] # e.g. [16, 32, 64, 128, 256]
        self.dilation_sizes = config['train']['dilation_sizes'] # e.g. [1,2,3,4]
        
        self.sequence_length = config['preprocess']['sequence_epochs']
        self.sleep_epoch_length = config['sleep_epoch_len']

        # Encoder U-units and reduction convs
        self.encoder_u_units = nn.ModuleList()
        self.encoder_reduce_convs = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()

        current_in_channels = 1 # Initial input channel for EEG/EOG
        for i in range(5): # 5 encoder stages
            u_unit_out_channels = self.filters[i]
            # For encoder 5, Keras uses pooling_sizes[3] and u_depths[3]
            pool_idx = min(i, len(self.pooling_sizes) - 1)
            depth_idx = min(i, len(self.u_depths) - 1)

            u_unit = UEncoderBlock(current_in_channels, u_unit_out_channels, self.kernel_size,
                                        self.pooling_sizes[pool_idx], self.u_inner_filter, self.u_depths[depth_idx],
                                        self.padding, self.activation_name)
            self.encoder_u_units.append(u_unit)
            
            # Keras: Conv2D(int(uX.get_shape()[-1] * 0.5), (1,1), ...)
            # uX.get_shape()[-1] is u_unit_out_channels
            reduce_conv_out_channels = u_unit_out_channels // 2
            reduce_conv = nn.Conv2d(u_unit_out_channels, reduce_conv_out_channels, kernel_size=(1,1),
                                    padding=self._calculate_same_padding(1,1), # 'same' for (1,1) kernel is 0 padding
                                    ) # Activation applied after reduce_conv in Keras
            self.encoder_reduce_convs.append(nn.Sequential(reduce_conv, get_activation(self.activation_name)))
            
            current_in_channels = reduce_conv_out_channels # Input for next U-unit is output of pool
            if i < 4: # Pooling for first 4 encoders
                pool = nn.MaxPool2d(kernel_size=(self.pooling_sizes[i], 1))
                self.encoder_pools.append(pool)

        # MSE Blocks
        self.mse_blocks = nn.ModuleList()
        for i in range(5):
            # Input to MSE is the output of reduce_conv
            mse_in_channels = self.filters[i] // 2
            mse_out_channels = self.mse_filters[i]
            mse_block = MSEBlock(mse_in_channels, mse_out_channels, self.kernel_size,
                                      self.dilation_sizes, self.padding, self.activation_name)
            self.mse_blocks.append(mse_block)

        # Decoder U-units and reduction convs
        self.decoder_upsamples = nn.ModuleList()
        self.decoder_u_units = nn.ModuleList()
        self.decoder_reduce_convs = nn.ModuleList()

        current_decoder_in_channels_from_prev_stage = self.mse_filters[4]

        for i in range(4): # 4 decoder stages (d4, d3, d2, d1)
            encoder_stage_idx = 3 - i # Corresponds to u4, u3, u2, u1
            
            self.decoder_upsamples.append(UpsampleBlock())
            
            u_unit_in_channels_concat = current_decoder_in_channels_from_prev_stage + self.mse_filters[encoder_stage_idx]
            
            u_unit_out_channels = self.filters[encoder_stage_idx]
            
            # Pooling size and depth for decoder U-unit match corresponding encoder U-unit
            pool_idx = encoder_stage_idx
            depth_idx = encoder_stage_idx

            u_unit = UEncoderBlock(u_unit_in_channels_concat, u_unit_out_channels, self.kernel_size,
                                        self.pooling_sizes[pool_idx], self.u_inner_filter, self.u_depths[depth_idx],
                                        self.padding, self.activation_name)
            self.decoder_u_units.append(u_unit)

            reduce_conv_out_channels = u_unit_out_channels // 2
            reduce_conv = nn.Conv2d(u_unit_out_channels, reduce_conv_out_channels, kernel_size=(1,1),
                                    padding=self._calculate_same_padding(1,1))
            self.decoder_reduce_convs.append(nn.Sequential(reduce_conv, get_activation(self.activation_name)))
            
            current_decoder_in_channels_from_prev_stage = reduce_conv_out_channels


    def _calculate_same_padding(self, kernel_h, kernel_w, dilation_h=1, dilation_w=1):
        pad_h = (dilation_h * (kernel_h - 1)) // 2
        pad_w = (dilation_w * (kernel_w - 1)) // 2
        return (pad_h, pad_w)

    def forward(self, x):
        # --- Encoder Path ---
        encoder_outputs_reduced = [] # Store outputs after reduction conv for MSE
        current_x = x
        for i in range(5):
            u_out = self.encoder_u_units[i](current_x)
            reduced_u_out = self.encoder_reduce_convs[i](u_out)
            encoder_outputs_reduced.append(reduced_u_out)
            if i < 4:
                current_x = self.encoder_pools[i](reduced_u_out)
            else: # Last encoder stage has no pooling after it in this chain
                current_x = reduced_u_out 
        
        # --- MSE Path ---
        mse_outputs = []
        for i in range(5):
            mse_out = self.mse_blocks[i](encoder_outputs_reduced[i])
            mse_outputs.append(mse_out)

        # --- Decoder Path ---
        # mse_outputs are [u1_mse, u2_mse, u3_mse, u4_mse, u5_mse]
        # Decoder starts with u5_mse
        decoder_current_x = mse_outputs[4] # u5_mse

        for i in range(4): # d4, d3, d2, d1
            skip_connection = mse_outputs[3 - i]
            
            upsampled_x = self.decoder_upsamples[i](decoder_current_x, skip_connection)
            concat_x = torch.cat([upsampled_x, skip_connection], dim=1)
            
            u_out = self.decoder_u_units[i](concat_x)
            decoder_current_x = self.decoder_reduce_convs[i](u_out)
            
        decoder_signal = mse_outputs[4] # u5_mse
        
        # d4
        skip_d4 = mse_outputs[3] # u4_mse
        up_d4 = self.decoder_upsamples[0](decoder_signal, skip_d4)
        cat_d4 = torch.cat([up_d4, skip_d4], dim=1)
        d4_full = self.decoder_u_units[0](cat_d4)
        d4_reduced = self.decoder_reduce_convs[0](d4_full)
        decoder_signal = d4_reduced

        # d3
        skip_d3 = mse_outputs[2] # u3_mse
        up_d3 = self.decoder_upsamples[1](decoder_signal, skip_d3)
        cat_d3 = torch.cat([up_d3, skip_d3], dim=1)
        d3_full = self.decoder_u_units[1](cat_d3)
        d3_reduced = self.decoder_reduce_convs[1](d3_full)
        decoder_signal = d3_reduced

        # d2
        skip_d2 = mse_outputs[1] # u2_mse
        up_d2 = self.decoder_upsamples[2](decoder_signal, skip_d2)
        cat_d2 = torch.cat([up_d2, skip_d2], dim=1)
        d2_full = self.decoder_u_units[2](cat_d2)
        d2_reduced = self.decoder_reduce_convs[2](d2_full)
        decoder_signal = d2_reduced
        
        # d1 (final U-unit, no reduction after it for zpad)
        skip_d1 = mse_outputs[0] # u1_mse
        up_d1 = self.decoder_upsamples[3](decoder_signal, skip_d1) # Assuming upsample is defined for this
        cat_d1 = torch.cat([up_d1, skip_d1], dim=1)
        # The last U-unit in Keras is:
        # d1 = create_u_encoder(layers.concatenate([up1, u1_mse]), self.filters[0], ...)
        # This U-unit is self.decoder_u_units[3]
        # Its output channels should be self.filters[0]
        d1_final = self.decoder_u_units[3](cat_d1) # This should be the output before padding

        target_h = self.sequence_length * self.sleep_epoch_length
        current_h = d1_final.size(2)
        
        pad_needed_total_h = target_h - current_h
        if pad_needed_total_h < 0: # Should not happen if U-Net is symmetric
            # This can happen if pooling/upsampling doesn't perfectly align.
            # Keras might handle this by cropping or specific resize.
            # For now, let's assume target_h >= current_h
            # If it happens, we might need to F.interpolate d1_final to target_h
            print(f"Warning: d1_final height {current_h} is greater than target height {target_h}. Adjusting via interpolation.")
            d1_final = F.interpolate(d1_final, size=(target_h, d1_final.size(3)), mode='bilinear', align_corners=False)
            pad_top_h = 0
            pad_bottom_h = 0
        else:
            pad_top_h = pad_needed_total_h // 2
            pad_bottom_h = pad_needed_total_h - pad_top_h

        # PyTorch ZeroPad2d: (pad_left, pad_right, pad_top, pad_bottom)
        padding_layer = nn.ZeroPad2d((0, 0, pad_top_h, pad_bottom_h))
        zpad_out = padding_layer(d1_final)
        
        return zpad_out


# --- Main TwoStream SalientSleepNet Model ---
class TwoStreamSalientModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.padding = config['train']['padding']
        self.activation_name = config['train']['activation']
        self.kernel_size = config['train']['kernel_size'] # This is an int for 1D kernels
        self.filters = config['train']['filters'] # List, e.g., [16, 32, 64, 128, 256]
        
        self.sequence_length = config['preprocess']['sequence_epochs']
        self.sleep_epoch_length = config['sleep_epoch_len']

        self.branch_eeg = Branch(config)
        self.branch_eog = Branch(config)

        # Attention mechanism layers
        # Input to attention is 'merge' which has self.filters[0] channels (output of Branch)
        attention_in_channels = self.filters[0]
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        # Keras: Dense(C//4) then Dense(C) on (N,1,1,C) tensor. Equivalent to 1x1 Convs.
        self.se_conv1 = nn.Conv2d(attention_in_channels, attention_in_channels // 4, kernel_size=1)
        self.se_act1 = get_activation(self.activation_name) # Keras uses self.activation
        self.se_conv2 = nn.Conv2d(attention_in_channels // 4, attention_in_channels, kernel_size=1)
        self.se_act2 = nn.Sigmoid() # Keras uses 'sigmoid' for excitation

        # Final classification layers
        # Input to Reshape is 'x' (output of attention) with self.filters[0] channels
        # Keras Reshape((self.sequence_length, self.sleep_epoch_length, self.filters[0]))
        # PyTorch: view(N, C, H_seq, W_epoch)
        
        # Conv2D after reshape in Keras: filters[0] channels, (1,1) kernel, 'tanh' activation
        self.final_conv1 = nn.Conv2d(self.filters[0], self.filters[0], kernel_size=1) # padding='same' for 1x1 is 0
        self.final_act1 = nn.Tanh()
        
        # AveragePooling2D((1, self.sleep_epoch_length))
        self.final_pool = nn.AvgPool2d(kernel_size=(1, self.sleep_epoch_length))
        
        # Conv2D(5, (self.kernel_size, 1), padding=self.padding, activation='softmax')
        # Input channels to this is self.filters[0]
        final_conv2_padding_h = (self.kernel_size - 1) // 2 # for 'same' with kernel_size (self.kernel_size, 1)
        self.final_conv2 = nn.Conv2d(self.filters[0], 5, kernel_size=(self.kernel_size, 1),
                                     padding=(final_conv2_padding_h, 0))
        self.final_softmax = nn.Softmax(dim=1) # Softmax over the 5 channels

    def forward(self, x_eeg, x_eog):
        # Input x_eeg, x_eog: (N, 1, sequence_length * sleep_epoch_length, 1)
        
        stream1_out = self.branch_eeg(x_eeg) # Shape: (N, filters[0], seq_len*epoch_len, 1)
        stream2_out = self.branch_eog(x_eog) # Shape: (N, filters[0], seq_len*epoch_len, 1)

        # Fusion
        mul_streams = stream1_out * stream2_out
        merged_features = stream1_out + stream2_out + mul_streams

        # Attention (SE block style)
        se = self.gap(merged_features) # (N, filters[0], 1, 1)
        se = self.se_conv1(se)
        se = self.se_act1(se)
        se = self.se_conv2(se)
        se = self.se_act2(se) # (N, filters[0], 1, 1) with sigmoid activation

        attention_out = merged_features * se # Element-wise multiply (broadcasting)

        reshaped_attention = attention_out.view(attention_out.size(0), # N
                                                self.filters[0],       # C
                                                self.sequence_length,  # H_new (seq_len)
                                                self.sleep_epoch_length) # W_new (epoch_len)
        
        x = self.final_conv1(reshaped_attention)
        x = self.final_act1(x) # Tanh
        
        x = self.final_pool(x)
        
        out_logits = self.final_conv2(x)
        
        
        out_probs = self.final_softmax(out_logits) # (N, 5, sequence_length, 1)
        
        out_probs = out_probs.squeeze(-1) # (N, 5, sequence_length)
        out_probs = out_probs.permute(0, 2, 1) # (N, sequence_length, 5)

        return out_probs

if __name__ == '__main__':
    # Example Usage:
    # Define a sample configuration (mirroring hyperparameters.yaml structure)
    sample_config = {
        'sleep_epoch_len': 3000, # 30s * 100Hz
        'preprocess': {
            'sequence_epochs': 20
        },
        'train': {
            'filters': [16, 32, 64, 128, 256],
            'kernel_size': 5, # This is an int in Keras for (k,1) kernels
            'pooling_sizes': [10, 8, 6, 4], # For first 4 encoders
            'dilation_sizes': [1, 2, 3, 4],
            'activation': 'relu',
            'u_depths': [4, 4, 4, 4], # For first 4 U-Encoders
            'u_inner_filter': 16, # middle_layer_filter for UEncoderBlockTorch
            'mse_filters': [8, 16, 32, 64, 128], # Example, adjust as per original
            'padding': 'same'
        }
    }

    # Create the PyTorch model
    pytorch_model = TwoStreamSalientModel(sample_config)
    pytorch_model.eval() # Set to evaluation mode if not training

    # Create dummy input tensors
    batch_size = 2
    num_channels = 1 # EEG/EOG are single channel time series
    seq_len = sample_config['preprocess']['sequence_epochs']
    epoch_len_samples = sample_config['sleep_epoch_len']
    
    # PyTorch input shape: (N, C_in, H, W)
    # H = seq_len * epoch_len_samples
    # W = 1 (as kernels are (k,1))
    
    dummy_eeg = torch.randn(batch_size, num_channels, seq_len * epoch_len_samples, 1)
    dummy_eog = torch.randn(batch_size, num_channels, seq_len * epoch_len_samples, 1)

    # Forward pass
    with torch.no_grad(): # Disable gradient calculations for inference
        predictions = pytorch_model(dummy_eeg, dummy_eog)
    
    print("PyTorch Model Initialized.")
    print(f"Input EEG/EOG shape: {dummy_eeg.shape}")
    print(f"Output predictions shape: {predictions.shape}") # Expected: (batch_size, sequence_length, 5)

    