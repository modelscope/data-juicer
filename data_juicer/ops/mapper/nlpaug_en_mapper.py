from copy import deepcopy

from loguru import logger
from pydantic import PositiveInt

from data_juicer.utils.availability_utils import AvailabilityChecking

from ..base_op import OPERATORS, Mapper

OP_NAME = 'nlpaug_en_mapper'

with AvailabilityChecking(['nlpaug'], OP_NAME):
    import nlpaug.augmenter.char as nac
    import nlpaug.augmenter.word as naw
    import nlpaug.flow as naf
    from nlpaug.util import Action


@OPERATORS.register_module(OP_NAME)
class NlpaugEnMapper(Mapper):
    """Mapper to simply augment samples in English based on nlpaug library."""

    _batched_op = True

    def __init__(self,
                 sequential: bool = False,
                 aug_num: PositiveInt = 1,
                 keep_original_sample: bool = True,
                 delete_random_word: bool = False,
                 swap_random_word: bool = False,
                 spelling_error_word: bool = False,
                 split_random_word: bool = False,
                 keyboard_error_char: bool = False,
                 ocr_error_char: bool = False,
                 delete_random_char: bool = False,
                 swap_random_char: bool = False,
                 insert_random_char: bool = False,
                 *args,
                 **kwargs):
        """
        Initialization method. All augmentation methods use default parameters
        in default. We recommend you to only use 1-3 augmentation methods at a
        time. Otherwise, the semantics of samples might be changed
        significantly.

        :param sequential: whether combine all augmentation methods to a
            sequence. If it's True, a sample will be augmented by all opened
            augmentation methods sequentially. If it's False, each opened
            augmentation method would generate its augmented samples
            independently.
        :param aug_num: number of augmented samples to be generated. If
            `sequential` is True, there will be total aug_num augmented samples
            generated. If it's False, there will be (aug_num *
            #opened_aug_method) augmented samples generated.
        :param keep_original_sample: whether to keep the original sample. If
            it's set to False, there will be only generated texts in the final
            datasets and the original texts will be removed. It's True in
            default.
        :param delete_random_word: whether to open the augmentation method of
            deleting random words from the original texts. e.g. "I love LLM"
            --> "I LLM"
        :param swap_random_word: whether to open the augmentation method of
            swapping random contiguous words in the original texts. e.g. "I
            love LLM" --> "Love I LLM"
        :param spelling_error_word: whether to open the augmentation method of
            simulating the spelling error for words in the original texts. e.g.
            "I love LLM" --> "Ai love LLM"
        :param split_random_word: whether to open the augmentation method of
            splitting words randomly with whitespaces in the original texts.
            e.g. "I love LLM" --> "I love LL M"
        :param keyboard_error_char: whether to open the augmentation method of
            simulating the keyboard error for characters in the original texts.
            e.g. "I love LLM" --> "I ;ov4 LLM"
        :param ocr_error_char: whether to open the augmentation method of
            simulating the OCR error for characters in the original texts.
            e.g. "I love LLM" --> "I 10ve LLM"
        :param delete_random_char: whether to open the augmentation method of
            deleting random characters from the original texts. e.g. "I love
            LLM" --> "I oe LLM"
        :param swap_random_char: whether to open the augmentation method of
            swapping random contiguous characters in the original texts.
            e.g. "I love LLM" --> "I ovle LLM"
        :param insert_random_char: whether to open the augmentation method of
            inserting random characters into the original texts. e.g. "I love
            LLM" --> "I ^lKove LLM"
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

        self.aug_num = aug_num
        if aug_num >= 10:
            logger.warning(f'Relatively large augmentation number [{aug_num}]'
                           f' might generate large number of new samples and '
                           f'requires more memory and disk space.')
        self.sequential = sequential
        self.keep_original_sample = keep_original_sample

        aug_pipeline = []
        # word level
        if delete_random_word:
            aug_pipeline.append(naw.RandomWordAug(action=Action.DELETE))
        if swap_random_word:
            aug_pipeline.append(naw.RandomWordAug(action=Action.SWAP))
        if spelling_error_word:
            aug_pipeline.append(naw.SpellingAug())
        if split_random_word:
            aug_pipeline.append(naw.SplitAug())

        # char level
        if keyboard_error_char:
            aug_pipeline.append(nac.KeyboardAug())
        if ocr_error_char:
            aug_pipeline.append(nac.OcrAug())
        if delete_random_char:
            aug_pipeline.append(nac.RandomCharAug(action=Action.DELETE))
        if swap_random_char:
            aug_pipeline.append(nac.RandomCharAug(action=Action.SWAP))
        if insert_random_char:
            aug_pipeline.append(nac.RandomCharAug(action=Action.INSERT))

        if self.sequential:
            self.aug = naf.Sequential(aug_pipeline)
        else:
            self.aug = aug_pipeline

    def process(self, samples):
        # no augmentation methods are opened
        if len(self.aug) == 0:
            if self.keep_original_sample:
                return samples
            else:
                return {key: [] for key in samples}

        texts_to_aug = samples[self.text_key][0]  # batch_size = 1
        res_samples = deepcopy(samples)

        # get augmented texts
        if self.sequential:
            aug_texts = self.aug.augment(texts_to_aug, n=self.aug_num)
        else:
            # apply each aug method to generate several augmented texts
            aug_texts = []
            for aug_method in self.aug:
                aug_texts += aug_method.augment(texts_to_aug, n=self.aug_num)

        # add augmented samples to the batch with other replicate fields
        if self.keep_original_sample:
            res_samples[self.text_key] += aug_texts
        else:
            res_samples[self.text_key] = aug_texts
        # add other replicate fields
        for key in res_samples:
            if key != self.text_key:
                res_samples[key] = res_samples[key] * \
                                   len(res_samples[self.text_key])
        return res_samples
