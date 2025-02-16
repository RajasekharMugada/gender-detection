import sounddevice as sd
import numpy as np

class AudioDeviceTest:
    @staticmethod
    def list_devices():
        """List all available audio devices"""
        print("\n=== Available Audio Devices ===")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"\nDevice {i}:")
            print(f"Name: {device['name']}")
            print(f"Max Input Channels: {device['max_input_channels']}")
            print(f"Max Output Channels: {device['max_output_channels']}")
            print(f"Default Sample Rate: {device['default_samplerate']}")

        print(f"\nDefault input device: {sd.query_devices(kind='input')}")
        print(f"Default output device: {sd.query_devices(kind='output')}")

    @staticmethod
    def test_recording():
        """Record and playback a short audio clip"""
        duration = 3  # seconds
        sample_rate = 44100
        channels = 1
        
        print("\n=== Recording Test ===")
        print("Recording for 3 seconds...")
        
        # Normalize the recording to prevent clipping
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype=np.float32
        )
        sd.wait()  # Wait until recording is finished
        print("Recording finished!")
        
        # Normalize audio to prevent distortion
        recording = recording / np.max(np.abs(recording))
        
        # Play back the recording with increased volume
        print("Playing back recording...")
        try:
            # Get default output device info
            device_info = sd.query_devices(kind='output')
            print(f"Using output device: {device_info['name']}")
            
            # Play with slightly increased volume
            sd.play(recording * 0.8, sample_rate)
            sd.wait()
            print("Playback finished!")
        except Exception as e:
            print(f"Playback error: {str(e)}")

def main():
    try:
        tester = AudioDeviceTest()
        tester.list_devices()
        tester.test_recording()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 