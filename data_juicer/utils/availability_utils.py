from loguru import logger

UNAVAILABLE_OPERATORS = {}
CHECK_SYSTEM_INFO_ONCE = True


class UnavailableOperator:

    def __init__(self, op_name, requires):
        self.op_name = op_name
        self.requires = requires

    def get_warning_msg(self):
        return f'This OP [{self.op_name}] is unavailable due to importing ' \
               f'third-party requirements of this OP failure: ' \
               f'{self.requires}. You can either run ' \
               f'`pip install -v -e .[sci]` to install all requirements for ' \
               f'all OPs, or run `pip install {" ".join(self.requires)}` ' \
               f'with library version specified by ' \
               f'`environments/science_requires.txt` to install libraries ' \
               f'required by this OP. Data processing will skip this OP later.'


class AvailabilityChecking:
    """Define a range that checks the availability of third-party libraries for
    OPs or other situations. If the checking failed, add corresponding OP to
    the unavailable OP
    list and skip them when initializing OPs with warnings.
    """

    def __init__(
        self,
        requires_list,
        op_name=None,
        requires_type=None,
    ):
        """
        Initialization method.

        :param requires_list: libraries to import in this range
        :param op_name: which op requires these libraries. In default, it's
            None, which means the importing process is not in an OP.
        """
        self.requires_list = requires_list
        self.op_name = op_name
        self.requires_type = requires_type

        self.error_msg = f'No module named {self.requires_list}. You might ' \
                         f'need to install it by running `pip install ' \
                         f'{" ".join(self.requires_list)}`.'
        if self.requires_type:
            self.error_msg += f' Or install all related requires by running ' \
                              f'`pip install -v -e .[{self.requires_type}]`'

    def __enter__(self):

        # only for python3.8 on mac
        global CHECK_SYSTEM_INFO_ONCE
        if CHECK_SYSTEM_INFO_ONCE:
            import os
            import platform
            import sys
            major, minor = sys.version_info[:2]
            system = platform.system()
            if major == 3 and minor == 8 and system == 'Darwin':
                logger.warning(
                    'The torch.set_num_threads function does not '
                    'work in python3.8 version on Mac systems. We will set '
                    'OMP_NUM_THREADS to 1 manually before importing torch')

                os.environ['OMP_NUM_THREADS'] = str(1)
                CHECK_SYSTEM_INFO_ONCE = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            if self.op_name:
                # ModuleNotFoundError for OP: register to UNAVAILABLE_OPERATORS
                UNAVAILABLE_OPERATORS[self.op_name] = UnavailableOperator(
                    op_name=self.op_name,
                    requires=self.requires_list,
                )
            else:
                # other situations: print error message and exit
                logger.error(f'{exc_type.__name__}: {exc_val}')
                logger.error(f'{exc_tb.tb_frame}')
                logger.error(self.error_msg)
                exit(0)
        elif exc_type is None:
            # import libs successfully
            pass
        else:
            # other exceptions: raise the exception directly
            return False

        # return True to suppress the exception
        return True
