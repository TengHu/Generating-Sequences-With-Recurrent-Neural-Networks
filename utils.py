import torch

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    
def one_hot(inputs, feature_size):
    '''
    One hot encoding with scatter function
    Input: (batch_size, sequence, ids)
    Output: (batch_size, sequence, features)
    '''
    batch_size = inputs.shape[0]
    sequence = inputs.shape[1]
    outputs = torch.zeros(batch_size*sequence, feature_size)
    index = inputs[:, :, 0].contiguous().view(-1, 1).long() # remove character dimension -> (batch_size*sequence, 1)
    # scatter(d, inputs, 1). inputs and outputs,  d!=dim size must equal, d==dim size could be different
        # in this case, passing value 1 to outputs(batch * sequence, feature)
    return outputs.scatter(1, index, 1).view(batch_size, sequence, -1)



