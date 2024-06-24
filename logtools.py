from abc import ABCMeta, abstractmethod
import os
from typing import Union, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class BaseLogger(metaclass=ABCMeta):
    def __init__(self):
        super(BaseLogger, self).__init__()
        
    @abstractmethod
    def log_parameters(self):
        pass
        
    @abstractmethod
    def log_metric(self, epoch: Optional[int] = None):
        pass
    
    def log_metrics(self, metrics_dict: dict, epoch: Optional[int] = None):
        for key, value in metrics_dict.items():
            self.log_metric(key, value, epoch)
            
    def log_stat_metric(self, name: str, values: np.ndarray, stat: List[str] = ["max"], epoch: Optional[int] = None):
        for s in stat:
            if s=="last":
                self.log_metric(f"{name} last", values[-1], epoch)
            else:
                self.log_metric(f"{name} {s}", getattr(values, s)(), epoch) # min, max, mean, std, sum
            
    def log_stat_metrics(self, metrics_dict: dict, stat: List[str] = ["max"], epoch: Optional[int] = None):
        for key, values in metrics_dict.items():
            self.log_stat_metric(key, values, stat, epoch)
    
    @abstractmethod
    def log_figure(self):
        pass
    
    
    
class EmptyLogger(BaseLogger):
    def __init__(self, project: Optional[str] = None, experiment_key: Optional[str] = None):
        super(EmptyLogger, self).__init__()
        
    def log_parameters(self, hparams: dict):
        pass
    
    def log_metric(self, name: str, value: Union[str, int, float], epoch: Optional[int] = None):
        pass
    
    def log_figure(self, name: str, fig: mpl.figure.Figure, overwrite: bool = False):
        pass

    def log_table(self, filename: str, tabular_data = None, headers: Union[bool, list] = False):
        pass

    def set_model_graph(self, model):
        pass
    
    def get_key(self):
        return None
    
    def end(self):
        pass


class CometLogger(BaseLogger):
    def __init__(self, project: Optional[str] = None, experiment_key: Optional[str] = None, **kwargs):
        super(CometLogger, self).__init__()
        import comet_ml
        if experiment_key is None:
            self.experiment = comet_ml.Experiment(project_name=project, parse_args=False, log_env_gpu=True, log_env_cpu=True, **kwargs)
        else:
            experiment_key = os.environ.get("COMET_experiment_key", experiment_key)
            record_on = False
            api = comet_ml.API() # Assumes API key is set in config/env
            api_experiment = api.get_experiment_by_key(experiment_key)
            self.experiment = comet_ml.ExistingExperiment(previous_experiment=experiment_key, parse_args=False, log_env_details=True, log_env_gpu=True, log_env_cpu=True, **kwargs)
        
        
    def log_parameters(self, hparams: dict):
        self.experiment.log_parameters(hparams) 
        
        
    def log_metric(self, name: str, value: Union[str, int, float], epoch: Optional[int] = None):
        self.experiment.log_metric(name, value, step=epoch)
        
            
    def log_figure(self, name: str, fig: mpl.figure.Figure, overwrite: bool = True):
        self.experiment.log_figure(name, fig, overwrite=overwrite)


    def log_table(self, filename: str, tabular_data = None, headers: Union[bool, list] = False):
        self.experiment.log_table(filename, tabular_data, headers=headers)


    def set_model_graph(self, model):
        self.experiment.set_model_graph(model)
        
        
    def get_metrics(self):
        metrics_list = self.experiment.get_metrics()
        metrics_dict = dict({})
        for m in metrics_list:
            if "sys." not in m["metricName"]:
                metrics_dict[m["metricName"]] = m["metricValue"]
        return metrics_dict
    
    
    def get_key(self):
        return self.experiment.get_key()
    
    
    def end(self):
        self.experiment.end()



class NeptuneLogger(BaseLogger):
    def __init__(self, project: Optional[str] = None, experiment_key: Optional[str] = None, **kwargs):
        super(NeptuneLogger, self).__init__()
        import neptune
        if experiment_key is None:
            self.run = neptune.init_run(project=project, source_files=["*.py"], **kwargs)
        else:
            self.run = neptune.init_run(project=project, source_files=["*.py"], with_id=experiment_key, **kwargs)
        
        
    def log_parameters(self, hparams: dict):
        self.run["parameters"] = hparams
        
        
    def log_metric(self, name: str, value: Union[str, int, float], epoch: Optional[int] = None):
        if epoch is not None:
            self.run["train/" + name].append(value=value, step=epoch)
        else:
            self.run["metrics/" + name] = value
        
            
    def log_figure(self, name: str, fig: mpl.figure.Figure, overwrite: bool = True):
        self.run["fig/" + name].upload(fig)


    def set_model_graph(self, model):
        self.run["model_graph"] = neptune.types.File.as_html(model)
        
        
    def get_metric(self, name: str, epoch: Optional[int] = None):
        if epoch is None:
            return self.run["metrics"][name].fetch_values()["value"]
        else:
            return self.run["train"][name].fetch_values()["value"]
        
        
    def get_metrics(self):
        return self.run.metrics.fetch()
    
    
    def get_key(self):
        return self.run.get_url().split("/")[-1]
    
    
    def end(self):
        self.run.stop()