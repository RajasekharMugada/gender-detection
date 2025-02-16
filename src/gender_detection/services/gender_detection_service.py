import numpy as np
import librosa
import torch
import torch.nn as nn

class GenderDetectionModel(nn.Module):
    def __init__(self):
        super(GenderDetectionModel, self).__init__()
        # Adjust model architecture for correct input dimensions
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Add adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        # Adjust fully connected layer dimensions
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 3)  # 3 classes: male, female, neutral
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class GenderDetectionService:
    def __init__(self):
        self.model = GenderDetectionModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.classes = ['male', 'female', 'neutral']
        
    async def preprocess_audio(self, audio_data: np.ndarray) -> torch.Tensor:
        """
        Preprocess the audio data for the model
        """
        if audio_data.size == 0:
            raise Exception("Empty audio data provided")
            
        if len(audio_data.shape) != 2 and len(audio_data.shape) != 1:
            raise Exception("Invalid audio data shape")
            
        try:
            # Ensure audio is flattened
            audio_flat = audio_data.flatten()
            
            if len(audio_flat) < 2048:  # Minimum length required for processing
                raise Exception("Audio data too short")
            
            # Extract mel spectrogram with fixed parameters
            mel_spec = librosa.feature.melspectrogram(
                y=audio_flat,
                sr=16000,
                n_mels=128,
                n_fft=2048,
                hop_length=512,
                win_length=2048
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_norm = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
            
            # Convert to tensor and add batch and channel dimensions
            tensor = torch.FloatTensor(mel_spec_norm).unsqueeze(0).unsqueeze(0)
            return tensor
            
        except Exception as e:
            raise Exception(f"Error preprocessing audio: {str(e)}")
        
    async def detect_gender(self, audio_data: np.ndarray) -> str:
        """
        Detect gender from audio data
        
        Args:
            audio_data (np.ndarray): Input audio data
            
        Returns:
            str: Detected gender (male/female/neutral)
        """
        if audio_data.size == 0:
            raise Exception("Empty audio data provided")
            
        try:
            # Preprocess audio
            tensor = await self.preprocess_audio(audio_data)
            tensor = tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(tensor)
                _, predicted = torch.max(outputs.data, 1)
                
            return self.classes[predicted.item()]
        except Exception as e:
            raise Exception(f"Error detecting gender: {str(e)}") 