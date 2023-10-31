
from datasets import Image

from data_juicer.utils.constant import DEFAULT_PREFIX

# A class to keep special tokens for multimodal information in the texts
# The tokens in this class can be updated by corresponding arguments in config
class SpecialTokens(object):
    # modality
    image = '<%s>' % (DEFAULT_PREFIX + 'image')

    # others
    eoc = f'<|{DEFAULT_PREFIX}eoc|>'

def load_images(paths):
    return [load_image(path) for path in paths]

def load_image(path):
    img_feature = Image()
    img = img_feature.decode_example(img_feature.encode_example(path))
    return img
