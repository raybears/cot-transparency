from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter


# For now these two formatters simply inherit from the unbiased formatters,
# because the dataexamples already have the bias inside them.
# this may change in the future
class ModelWrittenEvalsBiasedCOTFormatter(ZeroShotCOTUnbiasedFormatter):
    ...


class ModelWrittenEvalsBiasedFormatter(ZeroShotUnbiasedFormatter):
    ...
