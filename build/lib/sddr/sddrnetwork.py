import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
from .utils.utils import check_orthogonalization

## SDDR NETWORK PART
class SddrFormulaNet(nn.Module):
    '''
    This class represents an sddr network with a structured part, one or many deep models, a linear layer for 
    the structured part and a linear layer for the concatenated outputs of the deep models. The concatenated 
    outputs of the deep models are first filtered with an orthogonilization layer which removes any linear 
    parts of the deep output (by taking the Q matrix of the QR decomposition of the output of the structured part). 
    The two outputs of the linear layers are added so a prediction of a single parameter of the distribution is made
    and is returned as the final output of the network.
    The model follows the architecture described in Read me file.

    Parameters
    ----------
    deep_models_dict: dict
        dictionary where keys are names of the deep models and values are objects that define the deep models
    deep_shapes: dict
        dictionary where keys are names of the deep models and values are the outputs shapes of the deep models
    struct_shapes: int
        number of structural features
    orthogonalization_pattern: list of slice objects
        orthogonalization patterns for the deep neural networks, For each term in the design matrix wrt that the deep neural 
        network should be orthogonalized there is a slice in the list.
    p: float
        Dropout rate, probability of an element to be zeroed, the recommend value should between 0.01-0.1(depending on the feature numbers).
        The dropout is used for eastimate uncertainty.
    modify: bool, optional (default=True)
        If True, use the modified version with a correlation check when selecting slices for orthogonalization.
        If False, use the original version which concatenates all slices.
    
    Attributes
    ----------
    deep_models_dict: dict
        dictionary where keys are names of the deep models and values are objects that define the deep models
    orthogonalization_pattern: list of slice objects
        orthogonalization patterns for the deep neural networks
    structured_head: nn.Linear
        A linear layer which is fed the structured part of the data
    deep_head: nn.Linear
        A linear layer which is fed the unstructured part of the data
    deep_models_exist: Boolean
        This value is true if deep models have been used on init of the ssdr_single network, otherwise it is false
    '''
    
    def __init__(self, deep_models_dict, deep_shapes, struct_shapes, orthogonalization_pattern, p,
                 modify=True, ortho_manual = False, structured_bias=False, deep_bias=False):
        # modify for orthogonalization, deep_bias for bias in NN
        super(SddrFormulaNet, self).__init__()
        self.deep_models_dict = deep_models_dict
        self.deep_shapes = deep_shapes
        
        #register external neural networks
        for key, value in deep_models_dict.items():
            self.add_module(key,value) 
        
        self.orthogonalization_pattern = orthogonalization_pattern
        if struct_shapes == 0:
            self.structured_head = Zero_Layer()
        else:
            self.structured_head = nn.Linear(struct_shapes,1, bias = structured_bias)
        
        if len(deep_models_dict) != 0:
            output_size_of_deep_models  = sum([deep_shapes[key] for key in deep_shapes.keys()])
            self.deep_head = nn.Linear(output_size_of_deep_models,1, bias = deep_bias)
            self._deep_models_exist = True
        else:
            self._deep_models_exist = False
        
        self.p = p
        self.modify = modify
        self.ortho_manual = ortho_manual     
        
    def _orthog_layer(self, Q, Uhat):
        """
        Utilde = Uhat - QQTUhat
        """
        Projection_Matrix = Q @ Q.T
        Utilde = Uhat - Projection_Matrix @ Uhat
        
        return Utilde
    
    def _check_network_output_shape(self, Uhat_net, key,datadict):
        
        expected_batchsize = datadict[key].shape[0]
        expetec_output_size = self.deep_shapes[key]
        
        actual_output_shape = tuple(Uhat_net.shape)
        
        assert actual_output_shape == (expected_batchsize, expetec_output_size), f"Expected output of {key} to be {(expected_batchsize, expetec_output_size)} (batch-size, output_shape), but instead we found {actual_output_shape}"
    
    
    def forward(self, datadict,training=True):
        X = datadict["structured"]
        
        if self._deep_models_exist:

            Utilde_list = []
            latent_features_list = []  # collect latent features from each deep model
            
            for key in self.deep_models_dict.keys(): #assume that the input for the NN has the name of the NN as key
                net = self.deep_models_dict[key]
                Uhat_net = net(datadict[key])
                self._check_network_output_shape(Uhat_net, key, datadict)
                
                # orthogonalize the output of the neural network with respect to the parts of the structured part,
                # that contain the same input as the neural network
                if self.ortho_manual:
                    X_sliced_with_orthogonalization_pattern = torch.cat([X[:,sl] for sl in self.orthogonalization_pattern[key]],1)
                    Q, R = torch.qr(X_sliced_with_orthogonalization_pattern)
                    Utilde_net = self._orthog_layer(Q, Uhat_net)
                else:
                    if len(self.orthogonalization_pattern[key]) >0:
                        X_sliced_with_orthogonalization_pattern = torch.cat([X[:,sl] for sl in self.orthogonalization_pattern[key]],1)
                        Q, R = torch.qr(X_sliced_with_orthogonalization_pattern)
                        Utilde_net = self._orthog_layer(Q, Uhat_net)
                    else:
                        Utilde_net = Uhat_net
                # Check orthogonality between the structured part and the deep network output.
                # Here, X is assumed to contain both nonspline and (orthogonalized) spline terms,
                # and Utilde is the final output from the deep net after orthogonalization.
                # We convert both to numpy arrays.
                # structured_np = X #.detach().cpu().numpy()
                # deep_np = Utilde #.detach().cpu().numpy()
                # if not check_orthogonalization(structured_np, deep_np, tol=1e-8):
                #     print("Warning: The structured and deep network outputs are not orthogonal!")

                Utilde_list.append(Utilde_net)
                # Save the latent features from this deep branch
                latent_features_list.append(Utilde_net.detach())
                
            # Concatenate latent features from all deep models
            latent_features = torch.cat(latent_features_list, dim=1)    
            Utilde = torch.cat(Utilde_list, dim = 1) #concatenate the orthogonalized outputs of the deep NNs
            
            Utilde = nn.functional.dropout(Utilde,p=self.p,training=training)            
            deep_pred = self.deep_head(Utilde)
        else:
            latent_features = None
            deep_pred = 0
                
        X = nn.functional.dropout(X,p=self.p,training=training)        
        structured_pred = self.structured_head(X)
        
        pred = structured_pred + deep_pred
        
        
        return pred, latent_features
    # def forward(self, datadict, training=True):
    #     X = datadict["structured"]
        
    #     if self._deep_models_exist:
    #         Utilde_list = []
    #         latent_features_list = []  # collect latent features from each deep model
    #         for key in self.deep_models_dict.keys():
    #             net = self.deep_models_dict[key]
    #             Uhat_net = net(datadict[key])
    #             self._check_network_output_shape(Uhat_net, key, datadict)
                
    #             # Determine orthogonalization for the deep model output.
    #             if len(self.orthogonalization_pattern[key]) > 0:
    #                 if self.modify:
    #                     # Modified version: use only slices that are sufficiently correlated.
    #                     selected_slices = []
    #                     for sl in self.orthogonalization_pattern[key]:
    #                         X_slice = X[:, sl]
    #                         include_slice = False
    #                         # Check correlation between each column of X_slice and Uhat_net.
    #                         for i in range(X_slice.size(1)):
    #                             xi = X_slice[:, i]
    #                             for j in range(Uhat_net.size(1)):
    #                                 uj = Uhat_net[:, j]
    #                                 xi_mean = torch.mean(xi)
    #                                 uj_mean = torch.mean(uj)
    #                                 xi_std = torch.std(xi)
    #                                 uj_std = torch.std(uj)
    #                                 corr_val = torch.abs(torch.mean((xi - xi_mean) * (uj - uj_mean)) / (xi_std * uj_std + 1e-8))
    #                                 if corr_val > 0.5:
    #                                     include_slice = True
    #                                     break
    #                             if include_slice:
    #                                 break
    #                         if include_slice:
    #                             selected_slices.append(X_slice)
                        
    #                     if len(selected_slices) > 0:
    #                         X_sliced_with_orth_pattern = torch.cat(selected_slices, dim=1)
    #                         Q, R = torch.qr(X_sliced_with_orth_pattern)
    #                         Utilde_net = self._orthog_layer(Q, Uhat_net)
    #                     else:
    #                         Utilde_net = Uhat_net
    #                 else:
    #                     # Original behavior: use all slices.
    #                     X_sliced_with_orth_pattern = torch.cat([X[:, sl] for sl in self.orthogonalization_pattern[key]], dim=1)
    #                     Q, R = torch.qr(X_sliced_with_orth_pattern)
    #                     Utilde_net = self._orthog_layer(Q, Uhat_net)
    #             else:
    #                 Utilde_net = Uhat_net
                
    #             Utilde_list.append(Utilde_net)
    #             # Save the latent features from this deep branch
    #             latent_features_list.append(Utilde_net.detach())
            
    #         # Concatenate latent features from all deep models
    #         latent_features = torch.cat(latent_features_list, dim=1)
    #         Utilde = torch.cat(Utilde_list, dim=1)
    #         Utilde = nn.functional.dropout(Utilde, p=self.p, training=training)
    #         deep_pred = self.deep_head(Utilde)
    #     else:
    #         deep_pred = 0
    #         latent_features = None
        
    #     X = nn.functional.dropout(X, p=self.p, training=training)
    #     structured_pred = self.structured_head(X)
        
    #     pred = structured_pred + deep_pred
    #     return pred, latent_features
    
    def get_regularization(self, P):
        '''
        P = torch.from_numpy(P).float() # should have shape struct_shapes x struct_shapes, numpy array
        # do this somewhere else in the future?
        P = P.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        '''
        try:
            weights = self.structured_head.weight #should have shape 1 x struct_shapes
            regularization = weights @ P @ weights.T
        except:
            regularization = 0
        return regularization
        
        
class SddrNet(nn.Module):
    '''
    This class represents the full sddr network which can consist of one or many smaller sddr nets (in a parallel manner).
    Each smaller sddr predicts one distribution parameter and these are then sent into a transformation layer which applies
    constraints on the parameters depending on the given distribution. The output parameters are then fed into a distributional
    layer and a log-loss is computed. A regularization term is added to the log-loss to compute the total loss of the network.
    The model follows the architecture depicted here:
    https://docs.google.com/presentation/d/1cBgh9LoMNAvOXo2N5t6xEp9dfETrWUvtXsoSBkgVrG4/edit#slide=id.g8ed34c120e_5_16

    Parameters
    ----------
        family: Family 
            An instance of the class Family
        network_info_dict: dict
            A dictionary with keys being parameters of the distribution, e.g. "eta" and "scale"
            and values being dicts with keys deep_models_dict, deep_shapes, struct_shapes and orthogonalization_pattern
        p: float
            Dropout rate, probability of an element to be zeroed, the recommend value should between 0.01-0.1(depending on the feature numbers). The dropout is used for uncertainty-estimation.
            
    Attributes
    ----------
        family: Family 
            An instance of the class Family        
        single_parameter_sddr_list: dict
            A dictionary where keys are the name of the distribution parameter and values are the single_sddr object 
        distribution_layer_type: class object of some type of torch.distributions
            The distribution layer object, defined in the init and depending on the family, e.g. for
            family='normal' the object we will be of type torch.distributions.normal.Normal
        distribution_layer: class instance of some type of torch.distributions
            The final layer of the sddr network, which is initiated depending on the type of distribution (as defined 
            in family) and the predicted parameters from the forward pass
        latent_features: latent features from deep head
    '''
    
    def __init__(self, family, network_info_dict, p, modify, ortho_manual):
        super(SddrNet, self).__init__()
        self.family = family
        self.single_parameter_sddr_list = dict()
        for key, value in network_info_dict.items():
            deep_models_dict = value["deep_models_dict"]
            deep_shapes = value["deep_shapes"]
            struct_shapes = value["struct_shapes"]
            orthogonalization_pattern = value["orthogonalization_pattern"]
            self.single_parameter_sddr_list[key] = SddrFormulaNet(deep_models_dict, 
                                                                  deep_shapes, 
                                                                  struct_shapes, 
                                                                  orthogonalization_pattern,
                                                                  p,
                                                                  modify,
                                                                  ortho_manual)
            
            #register the SddrFormulaNet network
            self.add_module(key,self.single_parameter_sddr_list[key])
                
        self.distribution_layer_type = family.get_distribution_layer_type()
        self.latent_features = dict()
        
    def forward(self,datadict,training=True):
        """Performs a forward pass through the SDDR network.

        The forward pass computes predictions for each distribution parameter
        using the corresponding single-parameter SDDR network. These predictions
        are then transformed to satisfy distribution constraints, and a
        distribution layer is created based on the transformed parameters.

        Args:
            datadict (dict): A dictionary containing input data for each
                distribution parameter.
            training (bool, optional): Whether the network is in training mode.
                Defaults to True.

        Returns:
            tuple: A tuple containing the distribution layer and latent features.
        """
        self.regularization = 0
        pred = dict()

        for parameter_name, data_dict_param  in datadict.items():
            sddr_net = self.single_parameter_sddr_list[parameter_name]
            pred[parameter_name], self.latent_features[parameter_name] = sddr_net(data_dict_param,training=training)

        predicted_parameters = self.family.get_distribution_trafos(pred)
        
        self.distribution_layer = self.distribution_layer_type(**predicted_parameters)
        
        return self.distribution_layer, self.latent_features
    
    def get_log_loss(self, Y):
        ''' Compute log loss based on the trained distributional layer and the groundtruth Y '''
        log_loss = -self.distribution_layer.log_prob(Y)
        
        return log_loss
    
    def get_regularization(self, P):
        ''' Compute regularization given penalty matrix P '''
        regularization = 0
        for param  in self.single_parameter_sddr_list.keys():
            sddr_net = self.single_parameter_sddr_list[param]
            regularization += sddr_net.get_regularization(P[param])
        return regularization

    
class Zero_Layer(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Zero_Layer, self).__init__()
        self.weight = torch.tensor(0)

    def forward(self, input):
        return torch.tensor(0)
