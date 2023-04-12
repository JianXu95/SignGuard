# coding: utf-8
from .naive import *
from .lie import *
from .byzMean import *
from .max_sum import *
from .adaptive_attack import *

def attack(attack):

    attacks = {'random':random_attack,
               'sign_flip':signflip_attack,
               'zero':zero_attack,
               'noise':noise_attack,
               'nan':nan_attack,
               'label_flip':non_attack,
               'lie':little_is_enough_attack,
               'byzMean':byzMean_attack,
               'min_max':minmax_attack,
               'min_sum':minsum_attack,
               'adaptive_std':adaptive_attack_std,
               'adaptive_sign': adaptive_attack_sign,
               'adaptive_uv': adaptive_attack_uv,
               'non':non_attack
    }

    return attacks[attack]
