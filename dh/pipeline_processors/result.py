import os
import soundfile
import mimetypes
from diffusers.utils import export_to_video

class Result:
    def __init__(self, result, iteration):
        self.result = result
        self.iteration = iteration
        self.properties = iteration.get("result", {})

    def __repr__(self):
        return f"Result({self.result})"
    
    def get_raw_result(self):
        return self.result
    
    def get_output(self):
        if hasattr(self.result, "images"):
            return self.result.images[0]
        
        if hasattr(self.result, "image_embeds"):
            return self.result.image_embeds[0]

        if hasattr(self.result, "image_embeddings"):
            return self.result.image_embeddings[0]

        if hasattr(self.result, "frames"):
            return self.result.frames[0]
        
        if hasattr(self.result, "audios"):
            return self.result.audios[0].T.float().cpu().numpy()
        
        return None
    
    
    def get_output_list(self):
        if hasattr(self.result, "images"):
            return self.result.images
        
        if hasattr(self.result, "image_embeds"):
            return self.result.image_embeds

        if hasattr(self.result, "image_embeddings"):
            return self.result.image_embeddings

        if hasattr(self.result, "frames"):
            return self.result.frames
        
        if hasattr(self.result, "audios"):
            return [audio.T.float().cpu().numpy() for audio in self.result.audios]
            # return self.result.audios[0].T.float().cpu().numpy()
        
        return None
    
    # This method is used to guess the extension of the output file based on the content type.
    # It includes the leading "." in the extension.
    def guess_extension(self, content_type):
        ext = mimetypes.guess_extension(content_type)
        if ext is not None:
            return ext

        if content_type == "audio/wav":
            return ".wav"
        
        return ""

    def save(self, output_dir, default_base_name):
        content_type = self.properties.get("content_type", None)

        # do not save if no result content_type is specified
        if content_type is not None:
            file_base_name = self.properties.get("file_base_name", default_base_name)                          

            content_type = self.properties.get("content_type", "")
            extension = self.guess_extension(content_type)
            for i, output in enumerate(self.get_output_list()):
                output_path = os.path.join(output_dir, f"{file_base_name}-{i}{extension}")
                print(f"Saving result to {output_path}")
                if content_type.startswith("video"):
                    export_to_video(output, output_path, fps=self.properties.get("fps", 8))

                elif content_type.startswith("audio"):
                    soundfile.write(output_path, output, self.properties.get("sample_rate", 44100))

                elif hasattr(output, 'save'):
                    output.save(output_path)        

                else:
                    raise ValueError(f"Unsupported content type {content_type}")
