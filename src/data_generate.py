from scipy.stats import qmc
from src.physic_model import Helicopter
import pandas as pd

class Sample:
    def __init__(self, design_space, n_samples):
        self.bound = design_space
        self.n_samples = n_samples
        self.labels = list(design_space.keys())
        self.lower_bound = [self.bound[key][0] for key in self.labels]
        self.upper_bound = [self.bound[key][1] for key in self.labels]

    def CreateSample(self):
        sampler = qmc.LatinHypercube(d=len(self.labels))
        sample = sampler.random(n=self.n_samples)
        scaled_sample = qmc.scale(sample, self.lower_bound, self.upper_bound)
        return pd.DataFrame(scaled_sample, columns=self.labels)

class Dataset:
    def __init__(self, sample):
        self.sample = sample

    def CreateDataset(self):
        df = self.sample.CreateSample()

        sample_obj = Helicopter(df.to_dict('list'))

        Dp, Di = sample_obj.CalculateDrag()
        df['Total_drag'] = Dp + Di
        return df

def generate_dataset(design_space, n_samples):
    """
    Generate a dataset based on the provided design space and number of samples.

    Parameters:
    - design_space: A dictionary defining the bounds for each parameter.
    - n_samples: The number of samples to generate.

    Returns:
    - A pandas DataFrame containing the generated dataset with calculated drag values.
    """

    sample = Sample(design_space, n_samples)
    dataset = Dataset(sample)
    df_dataset = dataset.CreateDataset()
    return df_dataset
    