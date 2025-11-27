import os
import soundfile
import json
import mimetypes
import logging
from diffusers.utils import export_to_video, export_to_gif
from collections.abc import Iterable
from .security import validate_output_path, validate_string_input, SecurityError

logger = logging.getLogger("dw")

# Result saving constants
MAX_BASE_NAME_LENGTH = 200
DEFAULT_AUDIO_SAMPLE_RATE = 44100


class Result:
    """Manages and stores results from workflow steps.

    Handles result storage, artifact management, and file saving with support
    for multiple content types including images, video, audio, and JSON.
    """

    def __init__(self, result_definition):
        """Initialize Result with configuration for how to handle/save results.

        Args:
            result_definition: Dict containing result configuration including:
                - content_type: MIME type of the result
                - save: Boolean indicating if result should be saved
                - file_base_name: Base name for saved files
        """
        self.result_definition = result_definition
        self.result_list = []
        logger.debug(f"Initialized Result with definition: {result_definition}")

    def add_result(self, result):
        """Add one or more results to the result list.

        Args:
            result: Single result or list of results to store
        """
        if isinstance(result, list):
            logger.debug(f"Adding {len(result)} results to result list")
            self.result_list.extend(result)
        else:
            if isinstance(result, str):
                # Clean up string results by removing extra quotes and whitespace
                result = result.strip().strip('"').strip()
            logger.debug("Adding single result to result list")
            self.result_list.append(result)

    def get_artifacts(self):
        """Retrieve all artifacts from stored results.

        Returns:
            List of all artifacts from all results
        """
        artifacts = []
        for result in self.result_list:
            artifacts.extend(get_artifact_list(result))

        logger.debug(f"Retrieved {len(artifacts)} artifacts from results")
        return artifacts

    def get_artifact_properties(self, property_name):
        """Extract specific properties from results.

        Args:
            property_name: Name of property to extract from results

        Returns:
            List of property values from results where property exists
        """
        values = []
        for result in self.result_list:
            if isinstance(result, Iterable):
                if property_name in result:
                    values.append(result[property_name])
            else:
                logger.warning(
                    f"Skipping non-dict result when getting property {property_name}: {type(result)}"
                )

        logger.debug(f"Retrieved {len(values)} values for property: {property_name}")
        return values

    def save(self, output_dir, default_base_name):
        """Save results to files based on content type.

        Args:
            output_dir: Directory to save files in
            default_base_name: Default name to use for files
        """
        try:
            # Validate output directory
            validated_output_dir = validate_output_path(output_dir, None)
            validated_base_name = validate_string_input(
                default_base_name, max_length=MAX_BASE_NAME_LENGTH
            )

            # Add directory check/creation
            if not os.path.exists(validated_output_dir):
                logger.debug(f"Creating output directory: {validated_output_dir}")
                os.makedirs(validated_output_dir, exist_ok=True)
            elif not os.path.isdir(validated_output_dir):
                raise ValueError(
                    f"Output path exists but is not a directory: {validated_output_dir}"
                )
        except SecurityError as e:
            logger.error(f"Security validation failed for output: {e}")
            raise
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to create output directory: {e}")
            raise

        # Check if saving is enabled and content type is specified
        content_type = self.result_definition.get("content_type", None)
        if not self.result_definition.get("save", True) or content_type is None:
            logger.debug("Skipping save - disabled or no content type specified")
            return

        # Determine base filename with validation
        file_base_name = validated_base_name
        if "file_base_name" in self.result_definition:
            custom_base = validate_string_input(
                self.result_definition["file_base_name"],
                max_length=MAX_BASE_NAME_LENGTH,
            )
            file_base_name = custom_base + validated_base_name

        # Get file extension for content type
        extension = guess_extension(content_type)
        logger.debug(
            f"Saving with content type: {content_type}, extension: {extension}"
        )

        # Save each result
        for i, result in enumerate(self.result_list):
            if content_type.endswith("json"):
                # Handle JSON content type
                output_path = os.path.join(
                    validated_output_dir, f"{file_base_name}-{i}{extension}"
                )
                logger.info(f"Saving JSON result to {output_path}")
                with open(output_path, "w") as file:
                    file.write(json.dumps(result, indent=4))
            else:
                # Handle other content types
                for j, artifact in enumerate(get_artifact_list(result)):
                    self.save_artifact(
                        validated_output_dir,
                        artifact,
                        f"{file_base_name}-{i}.{j}",
                        content_type,
                        extension,
                    )

    def save_artifact(
        self, output_dir, artifact, file_base_name, content_type, extension
    ):
        """Save individual artifact to file based on its type.

        Args:
            output_dir: Directory to save file in
            artifact: The artifact to save
            file_base_name: Base name for the file
            content_type: MIME type of the content
            extension: File extension to use
        """
        if artifact is None:
            logger.warning(f"Skipping None artifact for {file_base_name}")
            return

        try:
            # Validate inputs
            validated_output_dir = validate_output_path(output_dir, None)
            validated_base_name = validate_string_input(
                file_base_name, max_length=MAX_BASE_NAME_LENGTH
            )
        except SecurityError as e:
            logger.error(f"Security validation failed for artifact save: {e}")
            raise

        if isinstance(artifact, dict):
            # Recursively save dictionary items
            logger.debug(
                f"Saving dictionary artifact with keys: {list(artifact.keys())}"
            )
            for k, v in artifact.items():
                self.save_artifact(
                    validated_output_dir,
                    v,
                    f"{validated_base_name}-{k}",
                    content_type,
                    extension,
                )
            return

        output_path = os.path.join(
            validated_output_dir, f"{validated_base_name}{extension}"
        )
        logger.info(f"Saving artifact to {output_path}")

        try:
            if content_type.startswith("video"):
                export_to_video(
                    artifact, output_path, fps=self.result_definition.get("fps", 8)
                )
            elif content_type == "image/gif":
                export_to_gif(
                    artifact, output_path, fps=self.result_definition.get("fps", 8)
                )
            elif content_type.startswith("audio"):
                soundfile.write(
                    output_path,
                    artifact,
                    self.result_definition.get(
                        "sample_rate", DEFAULT_AUDIO_SAMPLE_RATE
                    ),
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
        except Exception as e:
            logger.error(
                f"Error saving artifact to {output_path}: {str(e)}", exc_info=True
            )
            raise


def get_artifact_list(result):
    """Extract list of artifacts from a result object.

    Handles various result types including images, embeddings, frames, and audio.

    Args:
        result: Result object to extract artifacts from

    Returns:
        List of artifacts
    """
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


def guess_extension(content_type):
    """Determine file extension from MIME type.

    Args:
        content_type: MIME type string

    Returns:
        String containing file extension with leading dot
    """
    if not content_type:
        logger.warning("No content type provided for extension guess")
        return ""

    ext = mimetypes.guess_extension(content_type)
    if ext is not None:
        return ext

    # Handle special case for WAV files
    if content_type == "audio/wav":
        return ".wav"

    return ""
