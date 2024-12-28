import os
import soundfile
import json
import mimetypes
from diffusers.utils import export_to_video


class Result:
    def __init__(self, result_definition):
        self.result_definition = result_definition
        self.result_list = []

    def add_result(self, result):
        if isinstance(result, list):
            self.result_list.extend(result)
        else:
            self.result_list.append(result)

    def get_artifacts(self):
        artifacts = []
        for result in self.result_list:
            artifacts.extend(get_artifact_list(result))

        return artifacts

    def get_artifact_properties(self, property_name):
        values = []
        for result in self.result_list:
            if property_name in result:
                values.append(result[property_name])

        return values

    def save(self, output_dir, default_base_name):
        content_type = self.result_definition.get("content_type", None)

        # do not save if no result content_type is specified
        if content_type is not None:
            file_base_name = self.result_definition.get(
                "file_base_name", default_base_name
            )

            content_type = self.result_definition.get("content_type", "")
            extension = guess_extension(content_type)

            for i, result in enumerate(self.result_list):
                # if the result is meant to be saved as json, dump it to a file
                if content_type.endswith("json"):
                    output_path = os.path.join(
                        output_dir, f"{file_base_name}-{i}{extension}"
                    )
                    print(f"Saving result to {output_path}")
                    with open(output_path, "w") as file:
                        file.write(json.dumps(result, indent=4))

                else:
                    for j, artifact in enumerate(get_artifact_list(result)):
                        self.save_artifact(
                            output_dir,
                            artifact,
                            f"{file_base_name}-{i}.{j}",
                            content_type,
                            extension,
                        )

    def save_artifact(
        self, output_dir, artifact, file_base_name, content_type, extension
    ):
        if isinstance(artifact, dict):
            # here, if the result is a dictionary, we iterate over the items and save each one
            # by calling save_artifact recursively. which it will do until serializable properties are found
            for k, v in artifact.items():
                self.save_artifact(
                    output_dir, v, f"{file_base_name}-{k}", content_type, extension
                )

        else:
            output_path = os.path.join(output_dir, f"{file_base_name}{extension}")
            print(f"Saving result to {output_path}")

            if content_type.startswith("video"):
                export_to_video(
                    artifact, output_path, fps=self.result_definition.get("fps", 8)
                )

            elif content_type.startswith("audio"):
                soundfile.write(
                    output_path,
                    artifact,
                    self.result_definition.get("sample_rate", 44100),
                )

            elif content_type.endswith("json"):
                with open(output_path, "w") as file:
                    file.write(json.dumps(artifact, indent=4))

            elif content_type.startswith("text"):
                with open(output_path, "w") as file:
                    file.write(artifact)

            elif hasattr(artifact, "save"):
                artifact.save(output_path)

            else:
                raise ValueError(
                    f"Content type {content_type} does not match result type {type(artifact)}"
                )


def get_artifact_list(result):
    if hasattr(result, "images"):
        return result.images

    if hasattr(result, "image_embeds"):
        return result.image_embeds

    if hasattr(result, "image_embeddings"):
        return result.image_embeddings

    if hasattr(result, "frames"):
        return result.frames

    if hasattr(result, "audios"):
        return [audio.T.float().cpu().numpy() for audio in result.audios]

    if isinstance(result, list):
        return result

    return [result]


# This method is used to guess the extension of the output file based on the content type.
# It includes the leading "." in the extension.
def guess_extension(content_type):
    ext = mimetypes.guess_extension(content_type)
    if ext is not None:
        return ext

    if content_type == "audio/wav":
        return ".wav"

    return ""
