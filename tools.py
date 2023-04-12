import copy
import torch
import types
import math
import numpy as np

# --------------------------------------------------------------------- #
# Gradient access

def grad_of(tensor):
    """ Get the gradient of a given tensor, make it zero if missing.
    Args:
    tensor Given instance of/deriving from Tensor
    Returns:
    Gradient for the given tensor
    """
    # Get the current gradient
    grad = tensor.grad
    if grad is not None:
        return grad
    # Make and set a zero-gradient
    grad = torch.zeros_like(tensor)
    tensor.grad = grad
    return grad

def grads_of(tensors):
    """ Iterate of the gradients of the given tensors, make zero gradients if missing.
    Args:
    tensors Generator of/iterable on instances of/deriving from Tensor
    Returns:
    Generator of the gradients of the given tensors, in emitted order
    """
    return (grad_of(tensor) for tensor in tensors)

# ---------------------------------------------------------------------------- #
# "Flatten" and "relink" operations

def relink(tensors, common):
    """ "Relink" the tensors of class (deriving from) Tensor by making them point to another contiguous segment of memory.
    Args:
    tensors Generator of/iterable on instances of/deriving from Tensor, all with the same dtype
    common  Flat tensor of sufficient size to use as underlying storage, with the same dtype as the given tensors
    Returns:
    Given common tensor
    """
    # Convert to tuple if generator
    if isinstance(tensors, types.GeneratorType):
        tensors = tuple(tensors)
    # Relink each given tensor to its segment on the common one
    pos = 0
    for tensor in tensors:
        npos = pos + tensor.numel()
        tensor.data = common[pos:npos].view(*tensor.shape)
        pos = npos
    # Finalize and return
    common.linked_tensors = tensors
    return common

def flatten(tensors):
    """ "Flatten" the tensors of class (deriving from) Tensor so that they all use the same contiguous segment of memory.
    Args:
    tensors Generator of/iterable on instances of/deriving from Tensor, all with the same dtype
    Returns:
    Flat tensor (with the same dtype as the given tensors) that contains the memory used by all the given Tensor (or derived instances), in emitted order
    """
    # Convert to tuple if generator
    if isinstance(tensors, types.GeneratorType):
        tensors = tuple(tensors)
    # Common tensor instantiation and reuse
    common = torch.cat(tuple(tensor.view(-1) for tensor in tensors))
    # Return common tensor
    return relink(tensors, common)

# ---------------------------------------------------------------------------- #

def get_gradient(model):
    """ Get (optionally make each parameter's gradient) a reference to the flat gradient.
    Returns:
      Flat gradient (by reference: future calls to 'set_gradient' will modify it)
    """
    # Flatten (make if necessary)
    gradient = flatten(grads_of(model.parameters()))
    return gradient

def set_gradient(model, gradient):
    """ Overwrite the gradient with the given one.
    Args:
      gradient Given flat gradient
    """
    # Assignment
    grad_old = get_gradient(model)
    grad_old.copy_(gradient)

def get_gradient_values(model):
    """ Get (optionally make each parameter's gradient) a reference to the flat gradient.
    Returns:
      Flat gradient (by reference: future calls to 'set_gradient' will modify it)
    """

    gradient = torch.cat([torch.reshape(param.grad, (-1,)) for param in model.parameters()]).clone().detach()
    return gradient

def set_gradient_values(model, gradient):
    """ Overwrite the gradient with the given one.
    Args:
      gradient Given flat gradient
    """
    cur_pos = 0
    for param in model.parameters():
        param.grad = torch.reshape(torch.narrow(gradient, 0, cur_pos, param.nelement()), param.size()).clone().detach()
        cur_pos = cur_pos + param.nelement()

def get_parameter_values(model):
    """ Get (optionally make each parameter's gradient) a reference to the flat gradient.
    Returns:
      Flat gradient (by reference: future calls to 'set_gradient' will modify it)
    """

    parameter = torch.cat([torch.reshape(param.data, (-1,)) for param in model.parameters()]).clone().detach()
    return parameter
# ---------------------------------------------------------------------------- #
# Simple generator on the pairs (x, y) of an indexable such that index x < index y

def pairwise(data):
    """ Simple generator of the pairs (x, y) in a tuple such that index x < index y.
    Args:
    data Indexable (including ability to query length) containing the elements
    Returns:
    Generator over the pairs of the elements of 'data'
    """
    n = len(data)
    for i in range(n - 1):
        for j in range(i + 1, n):
            yield (data[i], data[j])

def elementwise_distance(data, point):
    '''
    :param data: tensor of gradients
    :param point: tensor of gradient
    :return: distance tensor
    '''
    return torch.norm(data-point,dim=1)

def pairwise_distance(data):
    device = data[0][0].device
    n = data.shape[0]
    distances = []
    for i in range(n):
        dist = elementwise_distance(data, data[i])
        distances.append(dist)
    return torch.stack(distances)


def pairwise_distance_faster(gradients):
    device = gradients[0][0].device
    n = gradients.shape[0]
    distances = torch.zeros((n,n),device=device)
    for gid_x, gid_y in pairwise(tuple(range(n))):
        dist = gradients[gid_x].sub(gradients[gid_y]).norm().item()
        if not math.isfinite(dist):
            dist = math.inf
        distances[gid_x][gid_y] = dist
        distances[gid_y][gid_x] = dist
    return distances


def max_pairwise_distance(data):
    '''
    :param data: tensor of gradients
    :return: max distance value
    '''
    n = data.shape[0]
    distances = [0] * (n * (n - 1) // 2)
    for i, (x, y) in enumerate(pairwise(tuple(range(n)))):
        dist = data[x].sub(data[y]).norm().item()
        if not math.isfinite(dist):
            dist = math.inf
        distances[i] = dist
    return max(distances)


def sum_distance(data, point):
    '''
    :param data: tensor
    :param point:
    :return:
    '''
    return elementwise_distance(data, point).sum().item()


def max_distance(data, point):
    '''
    :param data: tensor
    :param point:
    :return:
    '''
    return elementwise_distance(data, point).max().item()


def max_sum_distance(data):
    '''
    :param data: tensor
    :return:
    '''
    n = data.shape[0]
    dist_sum = [0]*n
    for i in range(n):
        dist_sum[i] = elementwise_distance(data, data[i]).sum().item()
    return max(dist_sum)


def pairwise_similarity(data):
    device = data[0][0].device
    n = data.shape[0]
    similarity = []
    for i in range(n):
        sim = torch.cosine_similarity(data, data[i], dim=-1)
        similarity.append(sim)
    return torch.stack(similarity)


def pairwise_similarity_faster(gradients):
    device = gradients[0][0].device
    n = gradients.shape[0]
    similarity = torch.zeros((n,n),device=device)
    for gid_x, gid_y in pairwise(tuple(range(n))):
        sim = torch.cosine_similarity(gradients[gid_x], gradients[gid_y], dim=-1)
        similarity[gid_x][gid_y] = sim
        similarity[gid_y][gid_x] = sim
    return similarity


def pairwise_sign_similarity(data):
    device = data[0][0].device
    n = data.shape[0]
    d = data.shape[1]
    similarity = []
    for i in range(n):
        sim = data.eq(data[i]).sum(dim=-1).float() - data.ne(data[i]).sum(dim=-1).float()
        similarity.append(sim/d)
    return torch.stack(similarity)


def pairwise_sign_similarity_plus(data):
    device = data[0][0].device
    n = data.shape[0]
    d = data.shape[1]
    similarity = []
    for i in range(n):
        pos_idx = torch.nonzero(data[i].eq(1.0)).squeeze()
        zero_idx = torch.nonzero(data[i].eq(0.0)).squeeze()
        neg_idx = torch.nonzero(data[i].eq(-1.0)).squeeze()
        sim_p = data[:,pos_idx].eq(data[i][pos_idx]).sum(dim=-1).float()
        sim_0 = data[:,zero_idx].eq(data[i][zero_idx]).sum(dim=-1).float()
        sim_n = data[:,neg_idx].eq(data[i][neg_idx]).sum(dim=-1).float()
        similarity.append((sim_p+sim_0+sim_n)/d)
    return torch.stack(similarity)

# -------------------------------------------------------------------------- #