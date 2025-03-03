import os
import yaml
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# torch imports
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
# pysddr imports
from .sddrnetwork import SddrNet, SddrFormulaNet
from .utils.dataset import SddrDataset
from .utils import checkups
from .utils.prepare_data import PrepareData
from .utils.family import Family
import warnings
import copy

class Sddr(object):
    '''
    The SDDR class is the main class the user interacts with in order to use the PySDDR framework. 
    This class includes functions to train the network, evaluate it, save/load its state, retrieve 
    coefficients, get the trained distribution, and make predictions.
    
    Parameters
    ----------
        **kwargs: either a list of parameters or a dict
            The user can give all the necessary parameters either one by one as variables or as a dictionary, 
            where the keys are the variables.
            
    Attributes
    -------
        config: dict
            A dictionary holding all the user-defined parameters.
        family: Family 
            An instance of the class Family.
        prepare_data: PrepareData object
            This object parses the formulas defined by the user and prepares the data.
        device: torch.device
            The current device, e.g. cpu or cuda.
        dataset: SddrDataset (torch.utils.data.Dataset)
            Loads and parses the input data.
        net: SddrNet 
            The SDDR network, composed of several smaller SddrFormulaNet modules.
        loader: DataLoader
            Loads data in batches.
        optimizer: torch optimizer
            The optimizer used during training.
    '''
    def __init__(self, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device:', self.device)
        
        # Depending on whether the user has given a dict as input or multiple arguments,
        # self.config should be a dict with keys the parameters defined by the user.
        for key in kwargs.keys():
            if key == 'config':
                self.config = kwargs['config']
            else:
                self.config = kwargs
            break
        
        # Create a family instance.
        self.family = Family(self.config['distribution'])
        self.p = self.config['train_parameters'].get('dropout_rate', 0)
        
        # Perform checks on the given distribution name, parameter names, and formulas.
        formulas = checkups(self.family.get_params(), self.config['formulas'])
        self.prepare_data = PrepareData(formulas,
                                        self.config['deep_models_dict'],
                                        self.config['train_parameters']['degrees_of_freedom'],
                                        self.config['modify'],
                                        self.config['ortho_manual'])
        
        # Set up the output directory.
        if 'output_dir' in self.config.keys():
            if not os.path.exists(self.config['output_dir']):
                os.mkdir(self.config['output_dir'])
        else:
            self.config['output_dir'] = './'
    
    def train(self, target, structured_data, unstructured_data=dict(), resume=False, plot=False):
        '''
        Trains the SddrNet for a specified number of epochs.
        '''
        epoch_print_interval = max(1, int(self.config['train_parameters']['epochs'] / 10))
        
        if resume:
            self.dataset = SddrDataset(structured_data, self.prepare_data, target, unstructured_data, fit=False)
        else:
            self.dataset = SddrDataset(structured_data, self.prepare_data, target, unstructured_data)
            self.net = SddrNet(self.family, self.# The code `prepare_data` is likely a function or
            # method call in Python, but without seeing the
            # actual implementation of the function, it is not
            # possible to determine exactly what it is doing. The
            # function name suggests that it is likely involved
            # in preparing or processing data in some way.
            prepare_data.network_info_dict, self.p, self.config['modify'], self.config['ortho_manual'])

            self.net = self.net.to(self.device)
            self.P = self.prepare_data.get_penalty_matrix(self.device)
            self._setup_optim()
            self.cur_epoch = 0
        
        val_split = self.config['train_parameters'].get('val_split', 0.2)

        n_val = int(len(self.dataset) * val_split)
        n_train = len(self.dataset) - n_val
        train, val = random_split(self.dataset, [n_train, n_val])
        
        self.train_loader = DataLoader(train, batch_size=self.config['train_parameters']['batch_size'])
        self.val_loader = DataLoader(val, batch_size=self.config['train_parameters']['batch_size'])
        train_loss_list = []
        val_loss_list = []
        
        if 'early_stop_epochs' in self.config['train_parameters'].keys():
            early_stop_counter = 0
            if not resume:
                self.cur_best_loss = sys.maxsize
            eps = self.config['train_parameters'].get('early_stop_epsilon', 0.001)
        
        print('Beginning training ...')
        for epoch in range(self.cur_epoch, self.config['train_parameters']['epochs']):
            self.net.train()
            self.epoch_train_loss = 0
            for batch in self.train_loader:
                target_batch = batch['target'].float().to(self.device)
                datadict_batch = batch['datadict']
                # Send each input batch to the device.
                for param in datadict_batch.keys():
                    for data_part in datadict_batch[param].keys():
                        datadict_batch[param][data_part] = datadict_batch[param][data_part].to(self.device)
                
                self.optimizer.zero_grad()
                output = self.net(datadict_batch)[0]  # only the distribution layer output
                loss = torch.mean(self.net.get_log_loss(target_batch))
                loss += self.net.get_regularization(self.P).squeeze_() 
                loss.backward()
                self.optimizer.step()
                self.epoch_train_loss += loss.item()
            
            self.epoch_train_loss /= len(self.train_loader)
            if epoch % epoch_print_interval == 0:
                print('Train Epoch: {} \t Training Loss: {:.6f}'.format(epoch, self.epoch_train_loss))
            train_loss_list.append(self.epoch_train_loss)
            
            with torch.no_grad():
                self.net.eval()
                self.epoch_val_loss = 0
                for batch in self.val_loader:
                    target_batch = batch['target'].float().to(self.device)
                    datadict_batch = batch['datadict']
                    for param in datadict_batch.keys():
                        for data_part in datadict_batch[param].keys():
                            datadict_batch[param][data_part] = datadict_batch[param][data_part].to(self.device)
                    _ = self.net(datadict_batch)[0]
                    val_batch_loss = torch.mean(self.net.get_log_loss(target_batch))
                    val_batch_loss += self.net.get_regularization(self.P).squeeze_() 
                    self.epoch_val_loss += val_batch_loss.item()
                if len(self.val_loader) != 0:
                    self.epoch_val_loss /= len(self.val_loader)
                val_loss_list.append(self.epoch_val_loss)
                
                if 'early_stop_epochs' in self.config['train_parameters'].keys():
                    dif = self.cur_best_loss - self.epoch_val_loss
                    if dif > eps:
                        self.cur_best_loss = self.epoch_val_loss
                        early_stop_counter = 0 
                    else:
                        early_stop_counter += 1
            if epoch % epoch_print_interval == 0 and len(self.val_loader) != 0:
                print('Train Epoch: {} \t Validation Loss: {:.6f}'.format(epoch, self.epoch_val_loss))
            if ('early_stop_epochs' in self.config['train_parameters'].keys() and 
                early_stop_counter == self.config['train_parameters']['early_stop_epochs']):
                print('Validation loss has not improved for the last {} epochs! Stopping training.'.format(early_stop_counter))
                break
        
        if plot:
            plt.figure()
            if plot == 'log':
                plt.plot(np.log(train_loss_list), label='train')
                if len(self.val_loader) != 0:
                    plt.plot(np.log(val_loss_list), label='validation')
            else:
                plt.plot(train_loss_list, label='train')
                if len(self.val_loader) != 0:
                    plt.plot(val_loss_list, label='validation')
            plt.legend(loc='upper left')
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.savefig(os.path.join(self.config['output_dir'], 'train_loss.png'))
            plt.show()
    
    def eval(self, param, bins=10, plot=True, data=None, get_feature=None):
        """
        Evaluates the trained SddrNet for a specific distribution parameter.
        """
        if data is None:
            data = self.dataset[:]["datadict"]
        if get_feature is None:
            get_feature = self.dataset.get_feature
        structured_head_params = self.net.single_parameter_sddr_list[param].structured_head.weight.detach().cpu()
        smoothed_structured = data[param]["structured"].cpu()
        list_of_spline_slices = self.prepare_data.dm_info_dict[param]['spline_info']['list_of_spline_slices']
        list_of_term_names = (self.prepare_data.dm_info_dict[param]['non_spline_info']['list_of_term_names'] +
                              self.prepare_data.dm_info_dict[param]['spline_info']['list_of_term_names'])
        list_of_spline_input_features = (self.prepare_data.dm_info_dict[param]['spline_info']['list_of_spline_input_features'])
        
        partial_effects = []
        can_plot = []
        xlabels = []
        ylabels = []
        
        for spline_slice, spline_input_features, term_name in zip(list_of_spline_slices, 
                                                                   list_of_spline_input_features, 
                                                                   list_of_term_names):
            if len(spline_input_features) == 1:
                feature = get_feature(spline_input_features[0])
                can_plot.append(True)
                ylabels.append(term_name)
                xlabels.append(spline_input_features[0])
            else:
                feature = []
                for feature_name in spline_input_features:
                    feature.append(get_feature(feature_name))
                can_plot.append(False)
                
            if self.p == 0:
                structured_pred = torch.matmul(smoothed_structured[:, spline_slice], structured_head_params[0, spline_slice])
                partial_effects.append((feature, structured_pred))
            else:
                structured_pred_dropout = []
                for _ in range(1000):
                    mask = torch.bernoulli(torch.full([1, structured_head_params.shape[1]], 1-self.p).float()).int()
                    structured_head_params_dropout = mask * structured_head_params
                    structured_pred = torch.matmul(smoothed_structured[:, spline_slice], structured_head_params_dropout[0, spline_slice]) * (1/(1-self.p))
                    structured_pred_dropout.append(structured_pred.numpy())
                structured_pred = np.mean(np.array(structured_pred_dropout), axis=0)
                ci950 = np.quantile(np.array(structured_pred_dropout), 0.025, axis=0)
                ci951 = np.quantile(np.array(structured_pred_dropout), 0.975, axis=0)
                ci250 = np.quantile(np.array(structured_pred_dropout), 0.25, axis=0)
                ci251 = np.quantile(np.array(structured_pred_dropout), 0.75, axis=0)
                partial_effects.append((feature, structured_pred, ci950, ci951, ci250, ci251))
            
        if plot:
            num_plots = sum(can_plot)
            if num_plots == 0:
                print('Nothing to plot. No (non-)linear partial effects specified for this parameter.')
            elif num_plots != len(partial_effects):
                print('Cannot plot', len(partial_effects) - num_plots, 'splines with more than one input.')
            for i in range(len(partial_effects)):
                if can_plot[i]:
                    if self.p == 0:
                        feature, partial_effect = partial_effects[i]
                        partial_effect = [x for _, x in sorted(zip(feature, partial_effect))]
                        plt.subplot(2,1,1)
                        plt.scatter(np.sort(feature), partial_effect)
                        plt.title('Partial effect %s' % (i+1))
                        plt.ylabel(ylabels[i])
                        plt.xlabel(xlabels[i])
                        plt.subplot(2,1,2)
                        plt.hist(feature, bins=bins)
                        plt.ylabel('Histogram of feature {}'.format(xlabels[i]))
                        plt.xlabel(xlabels[i])
                        plt.tight_layout()
                        plt.show()
                    else:
                        feature, partial_effect, ci950, ci951, ci250, ci251 = partial_effects[i]
                        re = np.array([[x, y, m, n, o] for _, x, y, m, n, o in sorted(zip(feature, partial_effect, ci950, ci951, ci250, ci251))])
                        partial_effect, ci950, ci951, ci250, ci251 = re[:,0], re[:,1], re[:,2], re[:,3], re[:,4]
                        plt.subplot(2,1,1)
                        plt.plot(np.sort(feature), partial_effect, label='Mean of partial_effect')
                        plt.fill_between(np.sort(feature), ci950, ci951, color='b', alpha=.1, label='95% confidence interval')
                        plt.fill_between(np.sort(feature), ci250, ci251, color='r', alpha=.2, label='50% confidence interval')
                        plt.legend()
                        plt.title('Partial effect %s' % (i+1))
                        plt.ylabel(ylabels[i])
                        plt.xlabel(xlabels[i])
                        plt.subplot(2,1,2)
                        plt.hist(feature, bins=bins)
                        plt.ylabel('Histogram of feature {}'.format(xlabels[i]))
                        plt.xlabel(xlabels[i])
                        plt.tight_layout()
                        plt.show()
        return partial_effects
    
    def save(self, name='model.pth'):
        state = {
            'epoch': self.config['train_parameters']['epochs'],
            'train_loss': self.epoch_train_loss,
            'val_loss': self.epoch_val_loss,
            'optimizer': self.optimizer.state_dict(),
            'sddr_net': self.net.state_dict()
        }
        warnings.simplefilter('always')
        warnings.warn("""Please note that the metadata for the structured input has not been saved. If you want to load the model and use
        it on new data you will need to also give the structured data used for training as input to the load function.""", stacklevel=2)
        
        save_path = os.path.join(self.config['output_dir'], name)
        torch.save(state, save_path)
        train_config_path = os.path.join(self.config['output_dir'], 'train_config.yaml')
        save_config = copy.deepcopy(self.config)
        save_config['train_parameters']['optimizer'] = str(self.optimizer)
        for net in save_config['deep_models_dict']:
            model = save_config['deep_models_dict'][net]['model']
            save_config['deep_models_dict'][net]['model'] = str(model)
        with open(train_config_path, 'w') as outfile:
            yaml.dump(save_config, outfile, default_flow_style=False)
    
    def _load_and_create_design_info(self, training_data, prepare_data):
        if isinstance(training_data, str):
            training_data = pd.read_csv(training_data, sep=None, engine='python')
        elif isinstance(training_data, pd.DataFrame):
            training_data = training_data
        prepare_data.fit(training_data)
    
    def _setup_optim(self):
        if 'optimizer' not in self.config['train_parameters'].keys():
            self.optimizer = optim.Adam(self.net.parameters())
            self.config['train_parameters']['optimizer_params'] = {'lr': 0.001,
                                                                    'betas': (0.9, 0.999),
                                                                    'eps': 1e-08,
                                                                    'weight_decay': 0,
                                                                    'amsgrad': False}
        else:
            if isinstance(self.config['train_parameters']['optimizer'], str):
                optimizer = eval(self.config['train_parameters']['optimizer'])
            else:
                optimizer = self.config['train_parameters']['optimizer']
            if 'optimizer_params' in self.config['train_parameters'].keys():
                self.optimizer = optimizer(self.net.parameters(), **self.config['train_parameters']['optimizer_params'])
            else:
                self.optimizer = optimizer(self.net.parameters())
    
    def load(self, model_name, training_data):
        self._load_and_create_design_info(training_data, self.prepare_data)
        self.P = self.prepare_data.get_penalty_matrix(self.device)
        if not torch.cuda.is_available():
            state_dict = torch.load(model_name, map_location='cpu')
        else:
            state_dict = torch.load(model_name)
        self.net = SddrNet(self.family, self.prepare_data.network_info_dict, self.p)
        self.net.load_state_dict(state_dict['sddr_net'])
        self.net = self.net.to(self.device)
        self._setup_optim()
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.cur_best_loss = state_dict['val_loss']
        self.cur_epoch = state_dict['epoch']
        print('Loaded model {} at epoch {} with a validation loss of {:.4f}'.format(model_name, self.cur_epoch, self.cur_best_loss))
    
    def coeff(self, param):
        list_of_slices = self.prepare_data.dm_info_dict[param]['non_spline_info']['list_of_non_spline_slices']
        list_of_slices += self.prepare_data.dm_info_dict[param]['spline_info']['list_of_spline_slices']
        list_of_term_names = self.prepare_data.dm_info_dict[param]['non_spline_info']['list_of_term_names']
        list_of_term_names += self.prepare_data.dm_info_dict[param]['spline_info']['list_of_term_names']
        all_coeffs = self.net.single_parameter_sddr_list[param].structured_head.weight.detach().cpu().numpy()
        coefs_dict = {}
        for term_name, slice_ in zip(list_of_term_names, list_of_slices):
            coefs_dict[term_name] = all_coeffs[0, slice_]
        return coefs_dict   
    
    def get_distribution(self):
        return self.net.distribution_layer
    
    def predict(self, data, unstructured_data=False, clipping=False, plot=False, bins=10):
        partial_effects = dict()
        predict_dataset = SddrDataset(data,
                                      prepare_data=self.prepare_data, 
                                      unstructured_data_info=unstructured_data,
                                      fit=False,
                                      clipping=clipping)
        
        datadict = predict_dataset[:]['datadict']
                
        for parameter in datadict.keys():
            for data_part in datadict[parameter].keys():
                datadict[parameter][data_part] = datadict[parameter][data_part].to(self.device)
                        
        with torch.no_grad():
            distribution_layer, latent_features = self.net(datadict, training=False)
            
        get_feature = lambda feature_name: data.loc[:, feature_name].values
        for param in datadict.keys():
            partial_effects[param] = self.eval(param, bins, plot, data=datadict, get_feature=get_feature)
        return distribution_layer, partial_effects, latent_features
    
if __name__ == "__main__":
    params = train()
