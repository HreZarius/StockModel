import numpy as np
import pandas as pd

def add_noise(data, noise_factor=0.01):
    """Добавляет небольшой шум к данным"""
    noise = np.random.normal(0, noise_factor, data.shape)
    return data + noise

def time_shift(data, shift_range=5):
    """Сдвигает временные ряды"""
    shift = np.random.randint(-shift_range, shift_range + 1)
    if shift > 0:
        return np.pad(data, ((shift, 0), (0, 0)), mode='edge')[:-shift]
    elif shift < 0:
        return np.pad(data, ((0, -shift), (0, 0)), mode='edge')[-shift:]
    return data

def scale_augmentation(data, scale_range=(0.95, 1.05)):
    """Масштабирует данные"""
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return data * scale_factor

def augment_sequence(sequence, augmentation_prob=0.5):
    """Применяет случайную аугментацию к последовательности"""
    if np.random.random() < augmentation_prob:
        aug_type = np.random.choice(['noise', 'shift', 'scale'])
        
        if aug_type == 'noise':
            return add_noise(sequence)
        elif aug_type == 'shift':
            return time_shift(sequence)
        elif aug_type == 'scale':
            return scale_augmentation(sequence)
    
    return sequence
