# tyee.model

**Built-in Model List**

| Class/Function Name                                | Functional Description                                       |
| -------------------------------------------------- | ------------------------------------------------------------ |
| [`NeuralTransformer`](#neuraltransformer)         | (LaBraM) A Transformer-based model for learning generic representations from extensive EEG data. |
| [`EEGTransformer`](#eegtransformer)               | A core Transformer feature extractor for patched, multi-channel EEG time-series data. |
| [`RelationAwareness`](#relationawareness)         | A module that computes a relationship matrix by fusing EEG, spatial location, and eye-tracking data. |
| [`EffNet`](#effnet)                               | A flexible 1D CNN backbone, inspired by EfficientNet, for extracting features from physiological signals. |
| [`BIOTEncoder`](#biotencoder)                       | A feature extractor that processes signals via per-channel STFT followed by a Transformer. |
| [`Conformer`](#conformer)                         | A hybrid Convolutional Neural Network (CNN) and Transformer model for EEG decoding. |
| [`EcgResNet34`](#ecgresnet34)                     | A 1D ResNet-34 architecture adapted for single-channel ECG classification. |
| [`MLSTM_FCN`](#mlstm_fcn)                         | A dual-branch hybrid model combining LSTM and a Fully Convolutional Network for multivariate time-series classification. |
| [`resnet18`](#resnet18)                         | A factory function that creates a `timm`-based ResNet-18 model for 2D image or spectrogram classification. |
| [`TwoStreamSalientModel`](#twostreamsalientmodel) | A two-stream, U-Net-like architecture for multi-modal sleep scoring using EEG and EOG. |
| [`AutoEncoder1D`](#autoencoder1d)                 | A 1D convolutional autoencoder with skip connections for signal reconstruction or translation tasks. |


## NeuralTransformer

LaBraM (Large Brain Model) is a powerful Transformer-based model designed for learning generic representations from extensive EEG data, making it highly suitable for various BCI (Brain-Computer Interface) tasks.

- **Paper:**[Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI](https://openreview.net/forum?id=QzTpTRVtrP)
- **Code Repository:**[LaBraM](https://github.com/935963004/LaBraM)

**Initialization Parameters**

The main `NeuralTransformer` class is initialized with the following parameters. Helper functions like `labram_base_patch200_200` provide pre-configured versions of this model.

- **`EEG_size`** (`int`): The total length (number of time points) of each EEG sample. Defaults to `1600`.
- **`patch_size`** (`int`): The size of each patch that the EEG time series is divided into. Defaults to `200`.
- **`in_chans`** (`int`): The number of input channels. If `in_chans=1`, `TemporalConv` is used as the embedding layer; otherwise, `PatchEmbed` is used. Defaults to `1`.
- **`out_chans`** (`int`): The number of output channels when using `TemporalConv`. Defaults to `8`.
- **`num_classes`** (`int`): The number of output classes for the final classification head. Defaults to `1000`.
- **`embed_dim`** (`int`): The embedding dimension (hidden size) of the Transformer model. Defaults to `200`.
- **`depth`** (`int`): The number of Transformer encoder layers. Defaults to `12`.
- **`num_heads`** (`int`): The number of heads in the multi-head attention mechanism. Defaults to `10`.
- **`mlp_ratio`** (`float`): The ratio of the MLP block's hidden dimension to `embed_dim`. Defaults to `4.0`.
- **`qkv_bias`** (`bool`): If `True`, adds a learnable bias to the q, k, v linear layers in the attention module. Defaults to `False`.
- **`drop_rate`** (`float`): The dropout probability for the MLP and projection layers. Defaults to `0.0`.
- **`attn_drop_rate`** (`float`): The dropout probability for the attention weights. Defaults to `0.0`.
- **`drop_path_rate`** (`float`): The stochastic depth decay rate. Defaults to `0.0`.
- **`use_abs_pos_emb`** (`bool`): If `True`, uses absolute positional embeddings. Defaults to `True`.

**Usage Example**

**Important Note**: This model expects a 4D input tensor with the shape `(B, N, A, T)`, which represents:

- `B`: Batch size
- `N`: Number of electrodes
- `A`: Number of patches each electrode's signal is divided into
- `T`: Patch size (number of time points per patch)

~~~python
# Instantiate a preset base model
# This model expects an input sample length of 800 (4 patches * 200 points/patch)
# and has 4 output classes
model = labram_base_patch200_200(
    EEG_size=800,
    num_classes=4
)
# Create a dummy EEG data tensor with the correct input format
# Batch size = 2, electrodes = 62, patches per electrode = 4, patch size = 200
dummy_eeg = torch.randn(2, 62, 4, 200)

# Pass the data through the model for a forward pass
output = model(dummy_eeg)
# Print the shape of the output tensor
# Expected output: torch.Size([2, 4]) (batch_size, num_classes)
print(output.shape)
~~~

[`Back to Top`](#tyeemodel)

## EEGTransformer

The `EEGTransformer` is a core feature extractor designed for Electroencephalography (EEG) signals. It employs a standard Transformer architecture, first dividing multi-channel time-series EEG data into patches, and then processing these patches through multiple Transformer encoder layers to learn deep temporal and spatial dependencies. It ultimately outputs a sequence of feature vectors representing the original EEG signal.

- **Paper:**[EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html)
- **Code Repository:**[EEGPT](https://github.com/BINE022/EEGPT)

**Initialization Parameters**

- **`img_size`** (`list`): A list of two integers, `[num_channels, sequence_length]`, representing the shape of the input EEG data. Defaults to `[64, 2560]`.
- **`patch_size`** (`int`): The size (number of time points) of each patch in the time dimension. Defaults to `64`.
- **`patch_stride`** (`int`, optional): The stride for the patch embedding layer. If `None`, the stride is equal to `patch_size`.
- **`embed_dim`** (`int`): The embedding dimension (hidden size) of the Transformer model. Defaults to `768`.
- **`embed_num`** (`int`): The number of summary tokens to use. Defaults to `1`.
- **`depth`** (`int`): The number of Transformer encoder layers. Defaults to `12`.
- **`num_heads`** (`int`): The number of heads in the multi-head attention mechanism. Defaults to `12`.
- **`mlp_ratio`** (`float`): The ratio of the MLP block's hidden dimension to `embed_dim`. Defaults to `4.0`.
- **`qkv_bias`** (`bool`): If `True`, adds a learnable bias to the q, k, v linear layers in the attention module. Defaults to `True`.
- **`drop_rate`** (`float`): The dropout probability for the MLP and projection layers. Defaults to `0.0`.
- **`attn_drop_rate`** (`float`): The dropout probability for the attention weights. Defaults to `0.0`.
- `drop_path_rate` (`float`): The stochastic depth decay rate. Defaults to `0.0`.
- **`norm_layer`** (`nn.Module`): The normalization layer to be used in the Transformer. Defaults to `nn.LayerNorm`.
- **`patch_module`** (`nn.Module`): The module used to convert the input signal into patch embeddings. Defaults to `PatchEmbed`.

**Usage Example**

**Important Note**: This model expects a 3D input tensor with the shape `(B, C, T)`, which represents:

- `B`: Batch size
- `C`: Number of channels/electrodes
- `T`: Sequence length (number of time points)

This model outputs feature representations, not final classification scores.

~~~python
# Define model parameters
in_channels = 62
sequence_length = 1024
patch_size = 64

# Calculate the number of patches
num_patches = sequence_length // patch_size

# Instantiate the model
model = EEGTransformer(
    img_size=[in_channels, sequence_length],
    patch_size=patch_size,
    embed_dim=512,
    depth=6,
    num_heads=8
)

# Create a dummy EEG data tensor with the correct input format
# Batch size = 2, channels = 62, sequence length = 1024
dummy_eeg = torch.randn(2, in_channels, sequence_length)

# Pass the data through the model for a forward pass to extract features
features = model(dummy_eeg)

# Print the shape of the output feature tensor
# Expected output: torch.Size([2, 16, 1, 512]) (batch_size, num_patches, embed_num, embed_dim)
print(features.shape)
~~~

[`Back to Top`](#tyeemodel)

## RelationAwareness

`RelationAwareness` is a module designed to compute relation-aware features. Its core function is to fuse EEG features, the spatial location information of electrodes, and eye-tracking data, and then use a self-attention mechanism to generate an adjacency matrix or attention map representing the relationships among these multimodal inputs.

- **Paper:**[Graph to Grid: Learning Deep Representations for Multimodal Emotion Recognition](https://dl.acm.org/doi/abs/10.1145/3581783.3612074)
- **Code Repository:**[G2G-ResNet18](https://github.com/Jinminbox/G2G)

**Initialization Parameters**

- **`head_num`** (`int`): The number of heads in the multi-head attention mechanism.
- **`input_size`** (`int`): The feature dimension of the input EEG signal per electrode.
- **`location_size`** (`int`): The dimension of the spatial coordinates for each electrode (e.g., 3 for 3D coordinates).
- **`expand_size`** (`int`): An internal expansion dimension used to embed the EEG, location, and eye-tracking data into a unified high-dimensional space.

**Usage Example**

**Important Note**: This module requires three types of input: EEG features, electrode locations, and eye-tracking features. Its output is a multi-head adjacency matrix representing the relationships between nodes.

~~~python
# --- 1. Define Model Parameters ---
head_num = 6
eeg_feature_dim = 5   # Corresponds to input_size
location_dim = 3      # Corresponds to location_size
expand_dim = 10       # Corresponds to expand_size
eye_feature_dim = 10  # Feature dimension for eye-tracking data

batch_size = 4
num_eeg_nodes = 62    # Number of EEG electrodes
num_eye_nodes = 6     # Number of eye-tracking data nodes

# --- 2. Instantiate the Model ---
model = RelationAwareness(
    head_num=head_num,
    input_size=eeg_feature_dim,
    location_size=location_dim,
    expand_size=expand_dim
)

# --- 3. Create Dummy Data with the Correct Input Format ---
# EEG Features: (batch_size, num_electrodes, feature_dim)
eeg_features = torch.randn(batch_size, num_eeg_nodes, eeg_feature_dim)
# Electrode Locations: (num_electrodes, coord_dim) -> expanded to match batch
electrode_locations = torch.randn(num_eeg_nodes, location_dim).expand(batch_size, -1, -1)
# Eye-tracking Features: (batch_size, num_eye_nodes, feature_dim)
eye_features = torch.randn(batch_size, num_eye_nodes, eye_feature_dim)

output_matrix = model(eeg_features, electrode_locations, eye_features)

~~~

[`Back to Top`](#tyeemodel)

## EffNet

`EffNet` is a deep convolutional neural network **backbone** designed for 1D physiological signals (like EEG, ECG), with an architecture inspired by EfficientNet. It efficiently extracts deep features from time-series data by stacking multiple Mobile Inverted Bottleneck blocks (`MBConv`). As a flexible feature extractor, its final fully connected layer can be replaced to suit various downstream tasks, such as classification, regression, or generating higher-dimensional embeddings.

- **Paper:**[SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity,ECG and Respiratory Signals](https://arxiv.org/abs/2405.17766)
- **Code Repository:**[SleepFM](https://github.com/rthapa84/sleepfm-codebase)

**Initialization Parameters**

- **`in_channel`** (`int`): The number of input channels for the signal (e.g., the number of EEG electrodes).
- **`num_additional_features`** (`int`): (Optional) The number of additional features to be concatenated before the final fully connected layer. If greater than 0, the model's forward pass expects a tuple `(x, additional_features)`. Defaults to `0`.
- **`depth`** (`list`): A list of integers defining the number of `Bottleneck` layers within each `MBConv` stage of the model. Defaults to `[1,2,2,3,3,3,3]`.
- **`channels`** (`list`): A list of integers defining the output channels for each stage of the network. Defaults to `[32,16,24,40,80,112,192,320,1280]`.
- **`dilation`** (`int`): The dilation rate for the first convolutional layer. Defaults to `1`.
- **`stride`** (`int`): The stride for the first convolutional layer. Defaults to `2`.
- **`expansion`** (`int`): The expansion factor inside the `MBConv` blocks. Defaults to `6`.

**Usage Example**

**Note**: The following example shows the direct usage of the `EffNet` class. By default, its final layer outputs a single value. However, in more complex applications , this `fc` layer is typically replaced, allowing `EffNet` to serve as a feature extraction backbone for different downstream tasks.

~~~python
# Define model parameters
in_channels = 22      # e.g., 22 EEG channels
sequence_length = 1000 # 1000 time points

# Instantiate the model (using default depth and channel configurations)
model = EffNet(
    in_channel=in_channels
)
# Create a dummy EEG data tensor with the correct input format
# Batch size = 4, channels = 22, sequence length = 1000
dummy_eeg = torch.randn(4, in_channels, sequence_length)
# Pass the data through the model for a forward pass
output = model(dummy_eeg)
# Print the shape of the output tensor
# Expected output: torch.Size([4, 1]) (batch_size, default_output_dim)
print(output.shape)
# --- How to use it as a feature extractor (Example) ---
# Replace the last layer
feature_dim = 512
model.fc = nn.Linear(model.fc.in_features, feature_dim)
# Now the forward pass will output 512-dimensional features
features = model(dummy_eeg)
# Expected output: torch.Size([4, 512])
print(features.shape)
~~~

[`Back to Top`](#tyeemodel)

## BIOTEncoder

The `BIOTEncoder`  is a feature extractor designed for multi-channel physiological signals (such as EEG). It works by transforming signals from each channel into the frequency domain and then utilizes a Transformer architecture to learn a feature vector that represents the entire multi-channel input.

- **Paper:**[BIOT: Cross-data Biosignal Learning in the Wild](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract-Conference.html)
- **Code Repository:**[BIOT](https://github.com/ycq091044/BIOT)

**Initialization Parameters**

- **`emb_size`** (`int`): The internal embedding dimension of the model. Defaults to `256`.
- **`heads`** (`int`): The number of heads in the Transformer's multi-head attention. Defaults to `8`.
- **`depth`** (`int`): The number of layers in the Transformer. Defaults to `4`.
- **`n_channels`** (`int`): The total number of learnable channel tokens. Should be greater than or equal to the actual number of channels in the data. Defaults to `16`.
- **`n_fft`** (`int`): The window size for the Short-Time Fourier Transform (STFT). Defaults to `200`.
- **`hop_length`** (`int`): The hop length for the STFT. Defaults to `100`.

**Usage Example**

**Important Note**: This model expects a 3D input tensor with the shape `(B, C, T)`, which represents:

- `B`: Batch size
- `C`: Number of signal channels
- `T`: Sequence length

The model outputs a fixed-dimensional feature vector.

~~~python
# Define model parameters
in_channels = 18
sequence_length = 2000
embedding_dimension = 256

# Instantiate the model
model = BIOTEncoder(
    emb_size=embedding_dimension,
    heads=8,
    depth=4,
    n_channels=in_channels, # Ensure n_channels >= actual channels
    n_fft=200,
    hop_length=100
)

# Create a dummy EEG data tensor with the correct input format
# Batch size = 4, channels = 18, sequence length = 2000
dummy_eeg = torch.randn(4, in_channels, sequence_length)

# Pass the data through the model for a forward pass to extract features
features = model(dummy_eeg)
~~~

[`Back to Top`](#tyeemodel)

## Conformer

The `Conformer` is a hybrid model designed specifically for EEG decoding, combining the advantages of Convolutional Neural Networks (CNNs) and Transformers. The model first uses a convolutional module to extract local spatio-temporal features and generate patch embeddings, which are then fed into a Transformer encoder to capture long-range dependencies.

- **Paper:** [EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization](https://ieeexplore.ieee.org/abstract/document/9991178/)
- **Code Repository:** [EEGConformer](https://github.com/eeyhsong/EEG-Conformer)

**Initialization Parameters**

- **`n_outputs`** (`int`): The number of output classes for the final classification layer.
- **`n_chans`** (`int`): The number of input EEG channels.
- **`n_times`** (`int`): The number of time points in the input signal. Must be provided if `final_fc_length="auto"`.
- **`n_filters_time`** (`int`): The number of temporal convolutional filters, which also defines the embedding size for the Transformer. Defaults to `40`.
- **`filter_time_length`** (`int`): The length of the temporal convolutional filters. Defaults to `25`.
- **`pool_time_length`** (`int`): The length of the temporal pooling kernel. Defaults to `75`.
- **`pool_time_stride`** (`int`): The stride for temporal pooling. Defaults to `15`.
- **`drop_prob`** (`float`): The dropout probability in the initial convolutional module. Defaults to `0.5`.
- **`att_depth`** (`int`): The number of layers in the Transformer encoder. Defaults to `6`.
- **`att_heads`** (`int`): The number of attention heads in the Transformer. Defaults to `10`.
- **`att_drop_prob`** (`float`): The dropout probability in the Transformer's attention layers. Defaults to `0.5`.
- **`final_fc_length`** (`int` or `str`): The input dimension of the final fully-connected layer. Can be set to `"auto"` for automatic calculation. Defaults to `"auto"`.
- **`return_features`** (`bool`): If `True`, the model returns the features just before the final classification layer. Defaults to `False`.
- **`activation`** (`nn.Module`): The activation function used in the convolutional module. Defaults to `nn.ELU`.
- **`activation_transfor`** (`nn.Module`): The activation function used in the Transformer's feed-forward network. Defaults to `nn.GELU`.

**Usage Example**

**Important Note**: This model expects a 3D input tensor with the shape `(B, C, T)`, which represents:

- `B`: Batch size
- `C`: Number of signal channels
- `T`: Sequence length

~~~python
# Define model parameters
n_outputs = 4
n_chans = 22
n_times = 1000

# Instantiate the model
model = Conformer(
    n_outputs=n_outputs,
    n_chans=n_chans,
    n_times=n_times,
    att_depth=6,
    att_heads=10,
    final_fc_length='auto'
)

# Create a dummy EEG data tensor with the correct input format
# Batch size = 8, channels = 22, sequence length = 1000
dummy_eeg = torch.randn(8, n_chans, n_times)

# Pass the data through the model for a forward pass
output = model(dummy_eeg)

# Print the shape of the output tensor
# Expected output: torch.Size([8, 4]) (batch_size, n_classes)
print(output.shape)
~~~

[`Back to Top`](#tyeemodel)

## EcgResNet34

`EcgResNet34` is a deep residual network designed for 1D Electrocardiogram (ECG) signals. It adapts the classic ResNet-34 architecture to a 1D format, learning features from time-series signals by stacking multiple residual blocks (`BasicBlock`). It is suitable for ECG classification tasks.

- **Paper:**[Diagnosis of Diseases by ECG Using Convolutional Neural Networks](https://www.hse.ru/en/edu/vkr/368722189)
- **Code Repository:**[ECGResNet34](https://github.com/lxdv/ecg-classification)

**Initialization Parameters**

- **`layers`** (`tuple`): A tuple of 4 integers that defines the number of residual blocks in each of the four stages of the ResNet. Defaults to `(1, 5, 5, 5)`.
- **`num_classes`** (`int`): The number of output classes for the final classification layer. Defaults to `1000`.
- **`zero_init_residual`** (`bool`): If `True`, the weights of the last BatchNorm layer in each residual block are initialized to zero. Defaults to `False`.
- **`groups`** (`int`): The number of groups for convolutions. `BasicBlock` only supports `1`. Defaults to `1`.
- **`width_per_group`** (`int`): The width per group. `BasicBlock` only supports `64`. Defaults to `64`.
- **`replace_stride_with_dilation`** (`list`, optional): A list of booleans that determines whether to replace strides with dilated convolutions in later stages.
- **`norm_layer`** (`nn.Module`): The normalization layer used in the model. Defaults to `nn.BatchNorm1d`.
- **`block`** (`nn.Module`): The type of residual block that makes up the network. Defaults to `BasicBlock`.

**Usage Example**

**Important Note**: This model expects a 3D input tensor with the shape `(B, 1, T)`, which represents:

- `B`: Batch size
- `C`: Number of signal channels (fixed to **1** for this model)
- `T`: Sequence length

~~~python
# Define model parameters
num_classes = 5
sequence_length = 2048

# Instantiate the model
# Using a layer configuration similar to ResNet-34
model = EcgResNet34(
    layers=(3, 4, 6, 3), # This is the standard configuration for ResNet-34
    num_classes=num_classes
)

# Create a dummy ECG data tensor with the correct input format
# Batch size = 8, channels = 1, sequence length = 2048
dummy_ecg = torch.randn(8, 1, sequence_length)

# Pass the data through the model for a forward pass
output = model(dummy_ecg)

# Print the shape of the output tensor
# Expected output: torch.Size([8, 5]) (batch_size, num_classes)
print(output.shape)

~~~

[`Back to Top`](#tyeemodel)

## AutoEncoder1D

The `AutoEncoder1D` is a convolutional autoencoder model designed for 1D time-series signals. It employs a classic Encoder-Decoder architecture and incorporates skip connections to help preserve and reconstruct detailed information. The model is suitable for tasks such as signal reconstruction, denoising, or translating one type of signal into another.

- **Paper:**[FingerFlex: Inferring Finger Trajectories from ECoG signals](https://arxiv.org/abs/2211.01960)
- **Code Repository:**[FingerFlex](https://github.com/Irautak/FingerFlex)

**Initialization Parameters**

- **`n_electrodes`** (`int`): The number of input electrodes/channels. Defaults to `30`.
- **`n_freqs`** (`int`): The number of features per channel (e.g., number of frequency bands after a wavelet or Fourier transform). Defaults to `16`.
- **`n_channels_out`** (`int`): The number of output channels from the decoder. Defaults to `21`.
- **`channels`** (`list`): A list of integers defining the number of output channels for each stage of the encoder. Defaults to `[8, 16, 32, 32]`.
- **`kernel_sizes`** (`list`): A list of integers defining the kernel size for each convolutional block in the encoder. Defaults to `[3, 3, 3]`.
- **`strides`** (`list`): A list of integers defining the downsampling stride for each convolutional block in the encoder. Defaults to `[4, 4, 4]`.
- **`dilation`** (`list`): A list of integers defining the dilation rate for each convolutional block in the encoder. Defaults to `[1, 1, 1]`.

**Usage Example**

**Important Note**: This model expects a 4D input tensor with the shape `(B, E, F, T)`, which represents:

- `B`: Batch size
- `E`: Number of electrodes/channels
- `F`: Number of features/frequencies per channel
- `T`: Sequence length

~~~python
# Define model parameters
n_electrodes = 62
n_freqs = 16
n_channels_out = 5  # e.g., decoding to 5 finger trajectories
sequence_length = 1024

# Instantiate the model
model = AutoEncoder1D(
    n_electrodes=n_electrodes,
    n_freqs=n_freqs,
    n_channels_out=n_channels_out,
    channels=[16, 32, 64],
    strides=[4, 4, 2]
)

# Create a dummy data tensor with the correct input format
# Batch size = 4, electrodes = 62, frequencies = 16, sequence length = 1024
dummy_input = torch.randn(4, n_electrodes, n_freqs, sequence_length)

# Pass the data through the model for a forward pass
output_signal = model(dummy_input)

# Print the shape of the output tensor
# Expected output: torch.Size([4, 5, 1024]) (batch_size, n_channels_out, original_sequence_length)
print(output_signal.shape)
~~~

[`Back to Top`](#tyeemodel)

## MLSTM_FCN

`MLSTM_FCN` is a deep learning model for multivariate time series classification. It employs a dual-branch hybrid architecture, combining the strengths of a Long Short-Term Memory (LSTM) network and a Fully Convolutional Network (FCN) to simultaneously capture long-term dependencies and local features from the time series.

- **Paper:**[Multivariate LSTM-FCNs for time series classification](https://www.sciencedirect.com/science/article/abs/pii/S0893608019301200)
- **Code Repository:**[MLSTM-FCN](https://github.com/titu1994/MLSTM-FCN)

**Initialization Parameters**

- **`max_nb_variables`** (`int`): The number of variables (i.e., channels) in the input multivariate time series.
- **`max_timesteps`** (`int`): The length (number of time points) of the input time series.
- **`nb_class`** (`int`): The number of output classes for the final classification layer.
- **`lstm_units`** (`int`): The number of hidden units in the LSTM layer. Defaults to `8`.
- **`dropout_rate`** (`float`): The dropout probability applied to the output of the LSTM path. Defaults to `0.8`.

**Usage Example**

**Important Note**: This model expects a 3D input tensor with the shape `(B, C, T)`, which represents:

- `B`: Batch size
- `C`: Number of variables/channels
- `T`: Sequence length

~~~python
# Define model parameters
num_variables = 5
num_timesteps = 640
num_classes = 9

# Instantiate the model
model = MLSTM_FCN(
    max_nb_variables=num_variables,
    max_timesteps=num_timesteps,
    nb_class=num_classes
)

# Create a dummy data tensor with the correct input format
# Batch size = 16, variables = 5, sequence length = 640
dummy_input = torch.randn(16, num_variables, num_timesteps)

# Pass the data through the model for a forward pass
output = model(dummy_input)

# Print the shape of the output tensor
# Expected output: torch.Size([16, 9]) (batch_size, num_classes)
print(output.shape)
~~~

[`Back to Top`](#tyeemodel)

## TwoStreamSalientModel

The `TwoStreamSalientModel` is a two-stream deep learning model designed for multi-modal physiological signals, particularly EEG and EOG. The core of this model is a complex U-Net-like architecture where each input stream (EEG and EOG) is processed by a separate, identical `Branch`. Each branch consists of an encoder-decoder path and a multi-scale feature extraction (MSE) module. Features from both branches are fused at a later stage and passed through an attention mechanism to produce the final sequence classification result, making it well-suited for tasks like sleep staging.

- **Paper:**[SalientSleepNet: Multimodal Salient Wave Detection Network for Sleep Staging](https://arxiv.org/abs/2105.13864)
- **Code Repository:**[SalientSleepNet](https://github.com/ziyujia/SalientSleepNet)

**Initialization Parameters**

The model is initialized with a single `config` dictionary, which contains all the hyperparameters that define the entire network architecture.

- `config`

   (

  ```
  dict
  ```

  ): A dictionary containing the model configuration. Its main keys include:

  - **`sleep_epoch_len`**: The length of a single sleep epoch in samples.

  - **`preprocess`**: Preprocessing-related configurations, such as `sequence_epochs` (the number of epochs in an input sequence).

  - `train`

    : Parameters related to training and network structure, for example:

    - `filters`: A list defining the number of channels at each stage of the U-Net.
    - `kernel_size`: The size of the convolutional kernels.
    - `pooling_sizes`: The sizes of the pooling layers in the encoder.
    - `dilation_sizes`: A list of dilation rates for the MSE module.
    - `u_depths`: The depth of the U-Net units.
    - `u_inner_filter`: The number of filters in the inner layers of the U-Net units.
    - `mse_filters`: The number of filters in the MSE modules.

**Usage Example**

**Important Note**: This model requires two separate input tensors, `x_eeg` and `x_eog`. The shape of each tensor should be `(B, 1, T, 1)`, which represents:

- `B`: Batch size
- `C`: Number of input channels (fixed to **1**)
- `T`: Total sequence length (`sequence_epochs * sleep_epoch_len`)
- `W`: Width dimension (fixed to **1**)

The model outputs a sequence of predictions, where each time step corresponds to a class probability distribution.

~~~python
# 1. Define a sample configuration dictionary
sample_config = {
    'sleep_epoch_len': 3000,
    'preprocess': {
        'sequence_epochs': 20
    },
    'train': {
        'filters': [16, 32, 64, 128, 256],
        'kernel_size': 5,
        'pooling_sizes': [10, 8, 6, 4],
        'dilation_sizes': [1, 2, 3, 4],
        'activation': 'relu',
        'u_depths': [4, 4, 4, 4],
        'u_inner_filter': 16,
        'mse_filters': [8, 16, 32, 64, 128],
        'padding': 'same'
    }
}

# 2. Instantiate the model
model = TwoStreamSalientModel(sample_config)

# 3. Create dummy data tensors with the correct input format
batch_size = 2
total_timesteps = sample_config['preprocess']['sequence_epochs'] * sample_config['sleep_epoch_len']
dummy_eeg = torch.randn(batch_size, 1, total_timesteps, 1)
dummy_eog = torch.randn(batch_size, 1, total_timesteps, 1)

# 4. Pass the data through the model for a forward pass
predictions = model(dummy_eeg, dummy_eog)

# 5. Print the shape of the output tensor
# Expected output: torch.Size([2, 20, 5]) (batch_size, sequence_length, num_classes)
# where num_classes is hardcoded to 5 in the model
print(predictions.shape)
~~~

[`Back to Top`](#tyeemodel)

## resnet18

This is a convenience factory function that creates a **ResNet-18** model using the `timm` library. The function simplifies the model instantiation process and supports loading pre-trained weights as well as custom checkpoints, making it well-suited for classification tasks on images or 2D representations like spectrograms.

- **Paper:** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Code Repository:** [pytorch-image-models (timm)](https://github.com/rwightman/pytorch-image-models)

**Function Parameters**

- **`num_classes`** (`int`): The number of output classes for the final classification layer.
- **`pretrained`** (`bool`): If `True`, loads weights pre-trained on ImageNet. Defaults to `True`.
- **`pretrained_cfg_overlay`** (`dict`, optional): A dictionary to override the default pre-trained model configuration from `timm` (e.g., to modify input size or mean/std). Defaults to `None`.
- **`checkpoint_path`** (`str`, optional): The path to a local checkpoint file (`.pth`). If provided, model weights will be loaded from this file. Defaults to `None`.

**Usage Example**

**Important Note**: The ResNet-18 model created by `timm` expects a 4D input tensor with the shape `(B, 3, H, W)`, which represents:

- `B`: Batch size
- `C`: Number of input channels (typically **3** for RGB images)
- `H`: Image height
- `W`: Image width

~~~python
# Define model parameters
num_classes = 10 # Assuming 10 gesture classes

# Call the function to create the ResNet-18 model
# pretrained=True will load ImageNet pre-trained weights
model = resnet18(
    num_classes=num_classes,
    pretrained=True
)

# Create a dummy data tensor with the correct input format
# Batch size = 4, channels = 3 (RGB), image size = 224x224
dummy_input = torch.randn(4, 3, 224, 224)

# Pass the data through the model for a forward pass
output = model(dummy_input)

# Print the shape of the output tensor
# Expected output: torch.Size([4, 10]) (batch_size, num_classes)
print(output.shape)
~~~

[`Back to Top`](#tyeemodel)
