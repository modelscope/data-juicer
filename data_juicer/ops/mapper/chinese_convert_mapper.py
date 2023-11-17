from ..base_op import OPERATORS, Mapper


def prepare_converter(mode):
    global OPENCC_CONVERTER
    import opencc
    OPENCC_CONVERTER = opencc.OpenCC(mode + '.json')


@OPERATORS.register_module('chinese_convert_mapper')
class ChineseConvertMapper(Mapper):
    """Mapper to convert Chinese between Traditional Chinese, Simplified Chinese
    and Japanese Kanji."""

    def __init__(self, mode: str = 's2t', *args, **kwargs):
        """
        Initialization method.

        :param mode: Choose the mode to convert Chinese,
        s2t: Simplified Chinese to Traditional Chinese,
        t2s: Traditional Chinese to Simplified Chinese,
        s2tw: Simplified Chinese to Traditional Chinese (Taiwan Standard),
        tw2s: Traditional Chinese (Taiwan Standard) to Simplified Chinese,
        s2hk: Simplified Chinese to Traditional Chinese (Hong Kong variant),
        hk2s: Traditional Chinese (Hong Kong variant) to Simplified Chinese,
        s2twp: Simplified Chinese to Traditional Chinese (Taiwan Standard)
               with Taiwanese idiom,
        tw2sp: Traditional Chinese (Taiwan Standard) to Simplified Chinese
               with Mainland Chinese idiom,
        t2tw: Traditional Chinese to Traditional Chinese (Taiwan Standard),
        tw2t: Traditional Chinese (Taiwan standard) to Traditional Chinese,
        hk2t: Traditional Chinese (Hong Kong variant) to Traditional Chinese,
        t2hk: Traditional Chinese to Traditional Chinese (Hong Kong variant),
        t2jp: Traditional Chinese Characters (Kyūjitai) to New Japanese Kanji,
        jp2t: New Japanese Kanji (Shinjitai) to Traditional Chinese Characters,
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        mode_list = [
            's2t', 't2s', 's2tw', 'tw2s', 's2hk', 'hk2s', 's2twp', 'tw2sp',
            't2tw', 'tw2t', 'hk2t', 't2hk', 't2jp', 'jp2t'
        ]
        assert mode in mode_list, 'Please make sure mode is one of {}'.format(
            mode_list)
        prepare_converter(mode)

    def process(self, sample):

        sample[self.text_key] = OPENCC_CONVERTER.convert(sample[self.text_key])
        return sample
