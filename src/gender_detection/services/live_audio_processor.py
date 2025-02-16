import asyncio
import sounddevice as sd
import numpy as np
from gender_detection.services.gender_detection_service import GenderDetectionService
import queue
import json

class LiveAudioProcessor:
    def __init__(self):
        self.service = GenderDetectionService()
        self.audio_queue = queue.Queue()
        self.result_queue = asyncio.Queue()
        self.running = True
        self.sample_rate = 16000
        self.chunk_duration = 3  # seconds
        self.channels = 1
        self.stream = None
        
    def audio_callback(self, indata, frames, time, status):
        """Callback function for the audio stream"""
        if status:
            print(f"Status: {status}")
        # Put the audio data in the queue
        self.audio_queue.put(indata.copy())
        
    async def process_audio(self):
        """Process audio chunks and predict gender"""
        while self.running:
            try:
                if not self.audio_queue.empty():
                    # Get audio chunk from queue
                    audio_data = self.audio_queue.get()
                    
                    # Convert audio data to the correct format
                    audio_data = audio_data.flatten().astype(np.float32)
                    
                    try:
                        # Detect gender
                        gender = await self.service.detect_gender(audio_data)
                        await self.result_queue.put(json.dumps({"gender": gender}))
                    except Exception as e:
                        await self.result_queue.put(json.dumps({"error": str(e)}))
                
                await asyncio.sleep(0.1)  # Small delay to prevent CPU overload
                
            except Exception as e:
                await self.result_queue.put(json.dumps({"error": str(e)}))

    async def get_next_result(self):
        """Get the next result from the queue"""
        return await self.result_queue.get()

    def stop_streaming(self):
        """Stop the audio stream"""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

    async def stream_results(self):
        """Stream results as Server-Sent Events"""
        try:
            # Start audio stream in background
            self.stream = sd.InputStream(
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * self.chunk_duration),
                callback=self.audio_callback
            )
            self.stream.start()

            # Start processing in background
            asyncio.create_task(self.process_audio())

            # Stream results
            while self.running:
                result = await self.get_next_result()
                yield f"data: {result}\n\n"
                await asyncio.sleep(0.1)
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            self.stop_streaming()

def main():
    async def run_processor():
        processor = LiveAudioProcessor()    
        async for result in processor.stream_results():
            print(result.strip())
    
    asyncio.run(run_processor())

if __name__ == "__main__":
    main() 