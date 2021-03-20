import numpy as np 
import pickle


def save_attention_weights(model, attention_mode, description, importance, run, float):
    """
    For saving attention layer weights from the
    `ICML2020 FilterWise Attention models`

    Notes:
    -----
        The weights directory is now hard coded at block4_pool. 
        When having a different attn layer position, need to make 
        this parameter more flexible.

    inputs:
    ------
        description: an individual category or a group category
    """
    ws = model.get_layer('att_layer_1').get_weights()[0]
    weights_path = 'attention_weights/block4_pool/attention={imp}/[hybrid]_{category}-imp{imp}-run{run}-float{float}.npy'.format(category=description, imp=round(importance,3), run=run, float=float)
    print('saving weights to: ', weights_path)
    np.save(weights_path, ws)
    print('attention weights saved.')


def save_fc2_weights(model, description, importance, run):
    """
    For saving the last layer's weights of a VGG16 trained 
    on various losses determined by attention intensity

    This is for cobb R1 response.
    """
    ws = model.get_layer('predictions').get_weights()
    print(ws[0].shape, ws[1].shape)

    importance = round(importance, 3)
    with open(f'attention_weights/cobb_lastLayer/attention={importance}/[hybrid]_{description}-imp{importance}-run{run}.pkl', 'wb') as f:
        pickle.dump(ws, f)
    
    print('fc2 weights saved.')