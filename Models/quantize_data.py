import numpy as np

def quantize_closing_price(features, targets):
    quantized_data = np.zeros(len(targets))
    for i,example in enumerate(features):
        if example <= targets[i]:
            quantized_data[i] = 1
        else:
            quantized_data[i] = 0

    return 
