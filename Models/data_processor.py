import numpy as np

class DataProcessor():
    def __init__(self,regularization=None):
        pass

    def quantize_closing_price(features, targets):
        quantized_data = np.zeros(len(targets))
        for i,example in enumerate(features):
            if example <= targets[i]:
                quantized_data[i] = 1
            else:
                quantized_data[i] = 0

        return quantized_data
