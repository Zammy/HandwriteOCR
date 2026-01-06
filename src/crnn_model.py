"""
CRNN Architecture for Handwriting OCR
======================================

Architecture Pipeline:
1. CNN Feature Extractor: Converts (B, 1, H, W) → (B, C, H', W')
2. Reshape to Sequence: (B, C, H', W') → (W', B, C*H')
3. BiLSTM: Models temporal dependencies along width
4. Linear Projection: Maps LSTM output to character probabilities
5. CTC Loss: Handles variable-length alignment

Input: Binary images of size (1, 128, 1024) - 1 channel, 128 height, 1024 width
Output: Character probabilities for CTC decoding
"""

import torch
import torch.nn as nn
from typing import Tuple


class CNNFeatureExtractor(nn.Module):
    """
    Extracts visual features from input images using convolutional layers.
    
    Architecture follows a typical CNN pattern:
    - Progressive channel expansion (1 → 64 → 128 → 256 → 512)
    - MaxPooling to reduce spatial dimensions
    - BatchNorm for training stability
    - ReLU activations
    
    Input:  (B, 1, 128, 1024)
    Output: (B, 512, 1, 64)  # Height reduced to 1, Width reduced by 16x
    """
    
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        
        # Block 1: 1 → 64 channels, 128x1024 → 64x512
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # /2 spatial dims
        )
        
        # Block 2: 64 → 128 channels, 64x512 → 32x256
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # /2 spatial dims
        )
        
        # Block 3: 128 → 256 channels, 32x256 → 16x128
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # /2 spatial dims
        )
        
        # Block 4: 256 → 256 channels, 16x128 → 8x64
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # /2 spatial dims
        )
        
        # Block 5: 256 → 512 channels, 8x64 → 4x64
        # Use (2,1) pooling to reduce height but preserve width
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # /2 height only
        )
        
        # Block 6: 512 → 512 channels, 4x64 → 2x64
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # /2 height only
        )
        
        # Block 7: 512 → 512 channels, 2x64 → 1x64
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # /2 height only
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN feature extractor.
        
        Args:
            x: Input tensor (B, 1, 128, 1024)
            
        Returns:
            Feature maps (B, 512, 1, 64)
        """
        x = self.conv1(x)  # (B, 64, 64, 512)
        x = self.conv2(x)  # (B, 128, 32, 256)
        x = self.conv3(x)  # (B, 256, 16, 128)
        x = self.conv4(x)  # (B, 256, 8, 64)
        x = self.conv5(x)  # (B, 512, 4, 64)
        x = self.conv6(x)  # (B, 512, 2, 64)
        x = self.conv7(x)  # (B, 512, 1, 64)
        
        return x


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM layer for sequence modeling.
    
    Processes features in both forward and backward directions,
    then concatenates outputs. Essential for capturing context
    in both directions (e.g., reading left-to-right and right-to-left).
    
    Input:  (T, B, input_size)  where T is sequence length
    Output: (T, B, hidden_size)
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(BidirectionalLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=False  # Expects (T, B, C) format
        )
        
        # Project concatenated bidirectional outputs to output_size
        # BiLSTM outputs hidden_size*2 (forward + backward)
        self.linear = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through BiLSTM.
        
        Args:
            x: Input tensor (T, B, input_size)
            
        Returns:
            Output tensor (T, B, output_size)
        """
        # LSTM returns (output, (h_n, c_n))
        lstm_out, _ = self.lstm(x)  # (T, B, hidden_size*2)
        
        # Project to output size
        output = self.linear(lstm_out)  # (T, B, output_size)
        
        return output


class CRNN(nn.Module):
    """
    Complete CRNN model for handwriting OCR.
    
    Architecture:
        Input (B, 1, 128, 1024)
            ↓
        CNN Feature Extractor
            ↓
        (B, 512, 1, 64)
            ↓
        Reshape to sequence: (64, B, 512)
            ↓
        BiLSTM Layer 1
            ↓
        BiLSTM Layer 2
            ↓
        Linear projection to character classes
            ↓
        Log Softmax
            ↓
        Output (64, B, num_classes) for CTC loss
    
    Args:
        num_classes: Number of output classes (characters + blank for CTC)
        cnn_output_height: Expected height after CNN (should be 1)
        lstm_hidden_size: Hidden size for LSTM layers
        lstm_num_layers: Number of stacked BiLSTM layers
    """
    
    def __init__(
        self,
        num_classes: int,
        cnn_output_height: int = 1,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2
    ):
        super(CRNN, self).__init__()
        
        self.num_classes = num_classes
        self.cnn_output_height = cnn_output_height
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # CNN Feature Extractor
        self.cnn = CNNFeatureExtractor()
        
        # After CNN: (B, 512, 1, 64)
        # We'll flatten height dimension: 512 * 1 = 512 features per timestep
        rnn_input_size = 512 * cnn_output_height
        
        # Stack multiple BiLSTM layers
        self.rnn_layers = nn.Sequential()
        
        # First BiLSTM layer
        self.rnn_layers.add_module(
            "bilstm_0",
            BidirectionalLSTM(
                input_size=rnn_input_size,
                hidden_size=lstm_hidden_size,
                output_size=lstm_hidden_size
            )
        )
        
        # Additional BiLSTM layers
        for i in range(1, lstm_num_layers):
            self.rnn_layers.add_module(
                f"bilstm_{i}",
                BidirectionalLSTM(
                    input_size=lstm_hidden_size,
                    hidden_size=lstm_hidden_size,
                    output_size=lstm_hidden_size
                )
            )
        
        # Final linear layer to project to character classes
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete CRNN model.
        
        Args:
            x: Input images (B, 1, H, W) where H=128, W=1024
            
        Returns:
            Log probabilities (T, B, num_classes) for CTC loss
            where T is the sequence length (width after CNN)
        """
        # 1. CNN Feature Extraction
        conv_out = self.cnn(x)  # (B, 512, 1, 64)
        
        # 2. Reshape for RNN: (B, C, H, W) → (W, B, C*H)
        # We want width to become the sequence dimension
        batch_size, channels, height, width = conv_out.size()
        
        # Reshape: (B, C, H, W) → (B, W, C*H)
        conv_out = conv_out.squeeze(2)  # (B, 512, 64) - remove height dim
        conv_out = conv_out.permute(2, 0, 1)  # (64, B, 512) - width becomes seq len
        
        # 3. BiLSTM sequence modeling
        rnn_out = self.rnn_layers(conv_out)  # (64, B, lstm_hidden_size)
        
        # 4. Project to character classes
        output = self.fc(rnn_out)  # (64, B, num_classes)
        
        # 5. Apply log softmax for CTC loss
        # CTC expects log probabilities
        output = torch.nn.functional.log_softmax(output, dim=2)
        
        return output
    
    def get_sequence_length(self, input_width: int = 1024) -> int:
        """
        Calculate the sequence length after CNN processing.
        
        Args:
            input_width: Input image width (default 1024)
            
        Returns:
            Sequence length after CNN pooling
        """
        # Track width through pooling layers
        # Pool1: /2, Pool2: /2, Pool3: /2, Pool4: /2
        # Pool5-7: width stays same (only height pooling)
        return input_width // 16  # 1024 / 16 = 64


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_shapes():
    """
    Test function to verify model shapes are correct.
    """
    print("=" * 70)
    print("CRNN Model Shape Testing")
    print("=" * 70)
    
    # Create model
    num_classes = 80  # Example: 26 lowercase + 26 uppercase + 10 digits + symbols + blank
    model = CRNN(
        num_classes=num_classes,
        cnn_output_height=1,
        lstm_hidden_size=256,
        lstm_num_layers=2
    )
    
    # Print model info
    print(f"\nModel created with {count_parameters(model):,} parameters")
    print(f"Number of output classes: {num_classes}")
    
    # Test with sample input
    batch_size = 4
    height = 128
    width = 1024
    
    x = torch.randn(batch_size, 1, height, width)
    print(f"\nInput shape: {x.shape}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Channels: 1 (grayscale)")
    print(f"  - Height: {height}")
    print(f"  - Width: {width}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"  - Sequence length: {output.shape[0]}")
    print(f"  - Batch size: {output.shape[1]}")
    print(f"  - Num classes: {output.shape[2]}")
    
    # Verify sequence length calculation
    expected_seq_len = model.get_sequence_length(width)
    actual_seq_len = output.shape[0]
    print(f"\nSequence length verification:")
    print(f"  - Expected: {expected_seq_len}")
    print(f"  - Actual: {actual_seq_len}")
    print(f"  - Match: {expected_seq_len == actual_seq_len} ✓" if expected_seq_len == actual_seq_len else "  - Match: False ✗")
    
    # Test CNN feature extractor separately
    print("\n" + "=" * 70)
    print("CNN Feature Extractor Testing")
    print("=" * 70)
    
    cnn = CNNFeatureExtractor()
    cnn.eval()
    with torch.no_grad():
        cnn_output = cnn(x)
    
    print(f"\nCNN input: {x.shape}")
    print(f"CNN output: {cnn_output.shape}")
    print(f"  - Batch: {cnn_output.shape[0]}")
    print(f"  - Channels: {cnn_output.shape[1]}")
    print(f"  - Height: {cnn_output.shape[2]} (should be 1)")
    print(f"  - Width: {cnn_output.shape[3]} (should be {width // 16})")
    
    print("\n" + "=" * 70)
    print("All shape tests passed! ✓")
    print("=" * 70)


def test_model_forward_backward():
    """
    Test forward and backward pass to ensure gradients flow correctly.
    """
    print("\n" + "=" * 70)
    print("CRNN Forward/Backward Pass Testing")
    print("=" * 70)
    
    # Create model
    num_classes = 80
    model = CRNN(num_classes=num_classes)
    
    # Create sample data
    batch_size = 2
    x = torch.randn(batch_size, 1, 128, 1024)
    
    # Forward pass
    print("\nRunning forward pass...")
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Create dummy target for CTC loss
    # Target lengths and input lengths
    target = torch.randint(1, num_classes, (batch_size, 30))  # Random targets
    input_lengths = torch.full((batch_size,), output.shape[0], dtype=torch.long)
    target_lengths = torch.randint(10, 31, (batch_size,), dtype=torch.long)
    
    # CTC Loss
    print("\nComputing CTC loss...")
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    loss = ctc_loss(output, target, input_lengths, target_lengths)
    print(f"Loss value: {loss.item():.4f}")
    
    # Backward pass
    print("\nRunning backward pass...")
    loss.backward()
    
    # Check gradients
    has_gradients = any(p.grad is not None for p in model.parameters())
    print(f"Gradients computed: {has_gradients} ✓" if has_gradients else f"Gradients computed: {has_gradients} ✗")
    
    # Check for NaN or Inf in gradients
    has_nan_inf = any(
        torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
        for p in model.parameters() if p.grad is not None
    )
    print(f"Gradients are finite: {not has_nan_inf} ✓" if not has_nan_inf else f"Gradients are finite: {not has_nan_inf} ✗")
    
    print("\n" + "=" * 70)
    print("Forward/backward test passed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    # Run all tests
    test_model_shapes()
    test_model_forward_backward()