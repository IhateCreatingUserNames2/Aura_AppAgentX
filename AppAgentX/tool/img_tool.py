import json
import os
import tempfile
from io import BytesIO
from typing import Dict, Union, List, IO
import numpy as np
import requests
from PIL import Image
from langchain_core.tools import tool
from scipy.optimize import linear_sum_assignment
from .. import config


def extract_features(
    image_inputs: Union[str, List[str], IO, List[IO]], model_name: str
):
    """
    Extract features from images, supporting single or batch image processing. Input can be file paths or file streams.

    Parameters:
        image_inputs: str, list, IO or list, image path, list of paths, file stream, or list of file streams
        model_name: str, name of the model to use

    Returns:
        dict: Feature data returned by the API
    """
    # Create temporary file list
    temp_files = []

    try:
        # Type check and preprocess
        is_single = True
        if isinstance(image_inputs, str):
            is_single = True
            inputs_list = [image_inputs]
        elif isinstance(image_inputs, (BytesIO, IO)):
            is_single = True
            inputs_list = [image_inputs]
        elif isinstance(image_inputs, List):
            is_single = False
            if not all(isinstance(x, (str, BytesIO, IO)) for x in image_inputs):
                raise TypeError(
                    "Elements in the list must be string paths or file stream objects"
                )
            inputs_list = image_inputs

        # Construct URL
        url = (
            f"{config.Feature_URI}/extract_single?model_name={model_name}"
            if is_single
            else f"{config.Feature_URI}/extract_batch?model_name={model_name}"
        )

        # Process input, convert stream to temporary file
        files = []
        for input_item in inputs_list:
            if isinstance(input_item, str):
                # If it's a file path, use it directly
                files.append(
                    ("files" if not is_single else "file", open(input_item, "rb"))
                )
            else:
                # If it's a file stream, create a temporary file
                temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                temp_files.append(
                    temp.name
                )  # Record temporary file path for later deletion

                # Ensure file pointer is at the start position
                if hasattr(input_item, "seek"):
                    input_item.seek(0)

                # Write data
                temp.write(input_item.read())

                # Reset stream position
                if hasattr(input_item, "seek"):
                    input_item.seek(0)
                temp.close()

                files.append(
                    ("files" if not is_single else "file", open(temp.name, "rb"))
                )

        # Send request
        response = requests.post(url, files=files)

        # Close all opened files
        for file in files:
            file[1].close()

        # Handle response
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    except Exception as e:
        raise Exception(f"Feature extraction failed: {str(e)}")

    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Failed to clean up temporary file: {str(e)}")


@tool
def element_similarity(
    page1: str,
    page2: str,
    json1: str,
    json2: str,
    feature_model: str = "resnet50",
    alpha: float = 0.7,
    beta: float = 0.3,
    distance_threshold: float = 0.5,
) -> Dict:
    """
    Calculate the similarity of elements in two pages.

    Parameters:
        page1: str, image path of the first page
        page2: str, image path of the second page
        json1: str, JSON file path of elements in the first page
        json2: str, JSON file path of elements in the second page
        feature_model: str, name of the feature extraction model
        alpha: float, weight of appearance features
        beta: float, weight of position features
        distance_threshold: float, matching threshold

    Returns:
        dict: Dictionary containing similarity scores and matching information
    """
    try:
        # 1. Load and preprocess data
        with open(json1, "r") as f:
            elements1 = json.load(f)
        with open(json2, "r") as f:
            elements2 = json.load(f)

        # 2. Crop and extract features
        features1 = _extract_element_features(page1, elements1, feature_model)
        features2 = _extract_element_features(page2, elements2, feature_model)

        # 3. Build distance matrix
        distance_matrix = _build_distance_matrix(
            features1, features2, elements1, elements2, alpha, beta
        )

        # 4. Use Hungarian algorithm for matching
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        # 5. Calculate final similarity and matching results
        matches = []
        total_similarity = 0
        valid_matches = 0

        for i, j in zip(row_ind, col_ind):
            distance = distance_matrix[i][j]
            if distance < distance_threshold:
                similarity = 1 - distance
                matches.append(
                    {
                        "element1": elements1[i],
                        "element2": elements2[j],
                        "similarity": float(similarity),
                    }
                )
                total_similarity += similarity
                valid_matches += 1

        # Calculate overall similarity
        overall_similarity = (
            total_similarity / valid_matches if valid_matches > 0 else 0.0
        )

        return {
            "similarity_score": float(overall_similarity),
            "matched_elements": valid_matches,
            "total_elements1": len(elements1),
            "total_elements2": len(elements2),
            "matches": matches,
            "status": "success",
            "message": "Successfully calculated element similarity",
        }

    except Exception as e:
        return {
            "similarity_score": 0.0,
            "status": "error",
            "message": f"Error occurred while calculating element similarity: {str(e)}",
        }


def element_img(page_path: str, elements_json: str, ID: int) -> BytesIO:
    """
    Crop elements from a page and return the image byte stream.

    Parameters:
        page_path: str, image path of the page
        elements_json: str, JSON string of elements
        ID: int, ID of the element to extract

    Returns:
        BytesIO: Cropped element image byte stream
    """
    try:
        # Load the original image
        image = Image.open(page_path)
        width, height = image.size

        # Parse JSON string
        elements = json.loads(elements_json)

        # Find the element with the specified ID
        target_element = None
        for element in elements:
            if element.get("ID") == ID:
                target_element = element
                break

        if target_element is None:
            raise ValueError(f"Element with ID {ID} not found")

        # Get and normalize the bounding box
        bbox = target_element["bbox"]
        x1 = max(0, int(bbox[0] * width))
        y1 = max(0, int(bbox[1] * height))
        x2 = min(width, int(bbox[2] * width))
        y2 = min(height, int(bbox[3] * height))

        # Ensure a valid cropping area
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid bounding box for element {ID}: ({x1}, {y1}, {x2}, {y2})"
            )

        # Crop the element
        element_image = image.crop((x1, y1, x2, y2))

        # Ensure the cropped image is not empty
        if element_image.size[0] == 0 or element_image.size[1] == 0:
            raise ValueError(f"Cropped image for element {ID} is empty")

        # Convert the image to a byte stream
        img_byte_arr = BytesIO()
        element_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)  # Move the pointer back to the start position

        return img_byte_arr

    except Exception as e:
        raise Exception(f"Error extracting element image: {str(e)}")


@tool
def elements_img(page_path: str, json_path: str, IDs: List[int]) -> List[BytesIO]:
    """
    Batch crop multiple elements from a page and return a list of image byte streams.

    Parameters:
        page_path: str, image path of the page
        json_path: str, JSON file path of elements
        IDs: List[int], list of element IDs to extract

    Returns:
        List[BytesIO]: List of cropped element image byte streams
    """
    try:
        # Load the original image
        image = Image.open(page_path)
        width, height = image.size

        # Load JSON file
        with open(json_path, "r") as f:
            elements = json.load(f)

        # Create a mapping from ID to element
        element_map = {element.get("ID"): element for element in elements}

        # Store results
        result_images = []

        # Process each requested ID
        for ID in IDs:
            try:
                if ID not in element_map:
                    print(f"Warning: Element with ID {ID} not found, skipping")
                    continue

                target_element = element_map[ID]

                # Get and normalize the bounding box
                bbox = target_element["bbox"]
                x1 = max(0, int(bbox[0] * width))
                y1 = max(0, int(bbox[1] * height))
                x2 = min(width, int(bbox[2] * width))
                y2 = min(height, int(bbox[3] * height))

                # Ensure a valid cropping area
                if x2 <= x1 or y2 <= y1:
                    print(
                        f"Warning: Invalid bounding box for element {ID}: ({x1}, {y1}, {x2}, {y2}), skipping"
                    )
                    continue

                # Crop the element
                element_image = image.crop((x1, y1, x2, y2))

                # Ensure the cropped image is not empty
                if element_image.size[0] == 0 or element_image.size[1] == 0:
                    print(f"Warning: Cropped image for element {ID} is empty, skipping")
                    continue

                # Convert the image to a byte stream
                img_byte_arr = BytesIO()
                element_image.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)  # Move the pointer back to the start position

                result_images.append(img_byte_arr)

            except Exception as e:
                print(f"Warning: Error processing element ID {ID}: {str(e)}")
                continue

        if not result_images:
            raise ValueError("No element images were successfully extracted")

        return result_images

    except Exception as e:
        raise Exception(f"Batch extraction of element images failed: {str(e)}")


def _extract_element_features(
    page_path: str, elements: List[Dict], feature_model: str
) -> List[np.ndarray]:
    """
    Crop elements from a page and extract features
    """
    try:
        # Load the original image
        image = Image.open(page_path)
        width, height = image.size
        # print(f"Processing image: {page_path}, size: {width}x{height}")

        features = []
        for i, element in enumerate(elements):
            try:
                # Get and normalize the bounding box
                bbox = element["bbox"]
                x1 = max(0, int(bbox[0] * width))
                y1 = max(0, int(bbox[1] * height))
                x2 = min(width, int(bbox[2] * width))
                y2 = min(height, int(bbox[3] * height))

                # Ensure a valid cropping area
                if x2 <= x1 or y2 <= y1:
                    print(f"Warning: Invalid bounding box for element {i}, skipping")
                    continue

                # Crop the element
                element_image = image.crop((x1, y1, x2, y2))

                # Ensure the cropped image is not empty
                if element_image.size[0] == 0 or element_image.size[1] == 0:
                    print(f"Warning: Cropped image for element {i} is empty, skipping")
                    continue

                # Create a temporary byte stream to store the cropped image
                img_byte_arr = BytesIO()
                element_image.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)

                # Extract features
                # print(f"Starting feature extraction for element {i}...")
                element_feature = extract_features(img_byte_arr, feature_model)
                # print(f"Feature extraction for element {i} succeeded")

                # Ensure the feature vector shape is correct
                feature_vector = np.array(element_feature["features"])
                if len(feature_vector.shape) > 1:
                    feature_vector = feature_vector.flatten()

                features.append(feature_vector)

            except Exception as e:
                print(f"Warning: Error processing element {i}: {str(e)}")
                print(f"Error type: {type(e)}")
                continue

        print(f"Successfully extracted features for {len(features)} elements")
        if not features:
            raise Exception("No element features were successfully extracted")

        return features

    except Exception as e:
        raise Exception(f"Error extracting element features: {str(e)}")


def _build_distance_matrix(
    features1: List[np.ndarray],
    features2: List[np.ndarray],
    elements1: List[Dict],
    elements2: List[Dict],
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Build a comprehensive distance matrix
    """
    n, m = len(features1), len(features2)
    distance_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            # Ensure the feature vector is one-dimensional
            f1 = features1[i].flatten()  # Flatten the feature vector
            f2 = features2[j].flatten()  # Flatten the feature vector

            # Calculate feature cosine distance
            norm1 = np.linalg.norm(f1)
            norm2 = np.linalg.norm(f2)

            if norm1 == 0 or norm2 == 0:
                feature_distance = (
                    1.0  # If the vector is a zero vector, set the maximum distance
                )
            else:
                # Calculate cosine similarity and convert to distance
                cosine_similarity = np.dot(f1, f2) / (norm1 * norm2)
                feature_distance = 1 - cosine_similarity

            # Calculate position distance
            pos_distance = _calculate_position_distance(
                elements1[i]["bbox"], elements2[j]["bbox"]
            )

            # Comprehensive distance
            distance_matrix[i][j] = alpha * feature_distance + beta * pos_distance

    return distance_matrix


def _calculate_position_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate the normalized position distance between two bounding boxes
    """
    # Calculate the center point
    center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
    center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]

    # Calculate the size
    size1 = [(bbox1[2] - bbox1[0]), (bbox1[3] - bbox1[1])]
    size2 = [(bbox2[2] - bbox2[0]), (bbox2[3] - bbox2[1])]

    # Calculate the center point distance
    center_distance = np.sqrt(
        (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
    )

    # Calculate the size difference
    size_difference = np.sqrt((size1[0] - size2[0]) ** 2 + (size1[1] - size2[1]) ** 2)

    # Normalize the total distance
    return (
        center_distance + size_difference
    ) / 4  # Divide by 4 because coordinates are in [0,1] range


if __name__ == "__main__":
    # Test batch element image extraction
    try:
        imgs = elements_img.invoke(
            {
                "page_path": "./log/screenshots/processed_images/labeled__step6_20250104_203251.png",
                "json_path": "./log/screenshots/processed_images/_step6_20250104_203251.json",
                "IDs": [0, 1, 2],
            }
        )
        print(f"Successfully extracted {len(imgs)} element images")
        # Optionally save images
        # for i, img in enumerate(imgs):
        #     with open(f"element_{i}.png", "wb") as f:
        #         f.write(img.getvalue())
    except Exception as e:
        print(f"Batch extraction of element images failed: {str(e)}")
