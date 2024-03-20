import base64
import imghdr
import mimetypes


def image_to_data_uri(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    image_type = imghdr.what(None, h=image_data)
    mime_type = mimetypes.types_map.get(f'.{image_type}')

    base64_encoded_data = base64.b64encode(image_data).decode('utf-8')

    data_uri = f'data:{mime_type};base64,{base64_encoded_data}'
    return data_uri
