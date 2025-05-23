"""Module"""
import math
import torch
import numpy as np
import pandas as pd

from torch import nn, optim
from torch.nn import init
import torch.nn.functional as F #funcoes de ativação



############################# Optimizer ##################################################################

class NewOptimizer(optim.AdamW):

    def __init__(self,params,lr=0.3, beta1=0.9, beta2=0.999,weight_decay=0.5):

        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay


        super().__init__(params,lr=self.lr,betas=(beta1, beta2),weight_decay=weight_decay)


    def calculate_alpha_bias(self, mean_bias, sd_bias, min_mean_bias, min_sd_bias, k):

        alpha_bias = (mean_bias + sd_bias) / (min_mean_bias + k * min_sd_bias)
        return alpha_bias


    def calculate_alpha_var(self, mean_var, sd_var, min_mean_var, min_sd_var, ro):

        alpha_var = (mean_var + sd_var) / (min_mean_var + 2* ro * min_sd_var)
        return alpha_var
    
    def prequential_error_with_fading(self, pred, true_label,preq_incorrect,preq_total,fading_factor=0.999):
        #Prequential error with fading factor for a batch of predictions"""

        preq_incorrect = fading_factor * preq_incorrect + int(pred != true_label)
        preq_total = fading_factor * preq_total + 1
        error = preq_incorrect / preq_total if preq_total != 0 else 0
        # Calculating accuracy from error
        accuracy_result = 1 - error
    
        return accuracy_result, preq_incorrect, preq_total


    def updates_hyperparameters(self, alpha_bias,alpha_var,opt):
        
        if alpha_bias >= 1:
            
           
            opt.param_groups[0]['weight_decay'] =  max(0,opt.param_groups[0]['weight_decay']/alpha_bias) # weight decay (lambda)
            opt.param_groups[0]['betas'] = (max(0,opt.param_groups[0]['betas'][0]/alpha_bias), # beta1
                                            max(0,opt.param_groups[0]['betas'][1]/alpha_bias)) # beta2
            opt.param_groups[0]['lr'] = min(1,opt.param_groups[0]['lr'] * alpha_bias) #eta
       

        elif alpha_var >= 1:
                
            opt.param_groups[0]['weight_decay'] =  min(1,opt.param_groups[0]['weight_decay'] * alpha_var)
            opt.param_groups[0]['betas'] = (min(1, opt.param_groups[0]['betas'][0] * alpha_var),
                                            min(1, opt.param_groups[0]['betas'][1] * alpha_var))
        
        


############################################ MLP ###############################################################

class MLP(nn.Module):
    """MLP with two-way line search"""

    def __init__(self, n_inputs, hidden_layer1, hidden_layer2, n_classes, device, **params):
        """define model elements"""
        super().__init__()
        self.__dict__.update(params)
        self.device = device
        self.out_act = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
        
        # MLP

        self.fc1 = nn.Linear(n_inputs,hidden_layer1)
        #self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        #self.out = nn.Linear(hidden_layer2, n_classes)
        self.out = nn.Linear(hidden_layer1, n_classes)
        # to gpu is available
        self.to(device)
    

    def forward(self, x):
        """Forward propagate input"""
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.out(x)
        
        return x


    def select_optimizer(self):
        
        if self.optimizer == "sgd":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                nesterov=self.nesterovs_momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                betas=(self.beta1, self.beta2),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adamw":
            optimizer = optim.AdamW(
                self.parameters(),
                betas=(self.beta1, self.beta2),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adadelta":
            optimizer = optim.Adadelta(
                self.parameters(),
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "rmsprop":
            optimizer = optim.RMSprop(
                self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                alpha=self.alpha,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "optim_adaptive":
            optimizer = NewOptimizer(self.parameters(),
                lr = self.lr,
                beta1 = self.beta1,
                beta2 = self.beta2,
                weight_decay = self.weight_decay,
            )
       
        return optimizer


    def sigmf(self,x,c,b):
        return 1 / (1 + np.exp(-c * (x - b))) 


    # recursive mean and standard 
    def meanstditer(self,miu_old,var_old,x,k):
        miu = miu_old + (x - miu_old)/k
        var = var_old + (x - miu_old)*(x - miu)
        std = np.sqrt(var/k)
        return miu, std, var


    def probit(self,miu,std): 
        p = (miu / (1 + math.pi * (std**2) / 8)**0.5)
        return p


    def train_evaluate(self,train_dl,mlp):
        """Test then train"""
        # define the optimization algorithm
        optimizer = mlp.select_optimizer()
        preds = torch.empty(0).to(self.device)
        true_labels = torch.empty(0).to(self.device)

        instance_number_mlp = 0
        update_hyperparameters_var_mlp = 0
        update_hyperparameters_bias_mlp = 0

        # for discriminative training ---> MLP
        
        miu_x_old = torch.from_numpy(np.zeros((1,self.fc1.in_features))) 
        var_x_old = torch.from_numpy(np.zeros((1,self.fc1.in_features))) 
        
        miu_var_old_mlp = 0
        var_var_old_mlp = 0
        miu_bias_old_mlp = 0
        var_bias_old_mlp = 0
        miumin_bias_mlp = 0   
        miumin_var_mlp = 0     
        stdmin_bias_mlp = 0   
        stdmin_var_mlp = 0     
        bias2_mlp_list = []
        var_mlp_list = []

        w_encoder = self.fc1.weight.clone()
        b_encoder = self.fc1.bias.clone().reshape(len(self.fc1.bias),1)
        w_mlp = mlp.out.weight.clone()
        b_mlp = mlp.out.bias.clone()

        
        for time_step, (inputs, targets, masks) in enumerate(train_dl):
  
            ################################  Test  #####################################
            
            # compute the MLP output
            yhat = self(inputs[-1])
            true_labels = torch.cat((true_labels, targets[-1].reshape(1)))


            ################################ Train ######################################
            yhat_train = self(inputs)
            optimizer.zero_grad()
            loss = mlp.criterion(yhat_train, targets)

                
            if mlp.optimizer == "optim_adaptive":

                instance_number_mlp = instance_number_mlp+1

                ## Incremental calculation of x_tail mean and 
                miu_x,std_x,var_x = self.meanstditer(miu_x_old,var_x_old,inputs[-1].cpu(),instance_number_mlp)
                miu_x_old = miu_x
                var_x_old = var_x


                # Expectation of z
                        
                py_mlp = self.probit(miu_x,std_x)
                py_t = torch.transpose(py_mlp, -1, 0)
                b_t = torch.transpose(b_encoder,-1,0)
             
                Ey_mlp = self.sigmf(w_encoder.cpu().detach().numpy().dot(py_t.detach().numpy()) + b_t.cpu().detach().numpy(),1,0) 
                Ez_mlp = w_mlp.cpu().detach().numpy().dot(Ey_mlp)
                Ez_mlp = np.exp(Ez_mlp)
                Ez_mlp = Ez_mlp/sum(Ez_mlp)        
                Ez2_mlp = w_mlp.cpu().detach().numpy().dot(Ey_mlp**2) 
                Ez2_mlp = np.exp(Ez2_mlp)
                Ez2_mlp = Ez2_mlp/sum(Ez2_mlp)

                # Network mean calculation
                        
                bias2_mlp = (Ez_mlp - targets[-1].cpu().numpy()) **2 
                ns_mlp    = bias2_mlp
                NS_mlp    = np.linalg.norm(ns_mlp,'fro')

                ## Incremental calculation of NS mean and variance
                miu_bias_mlp,std_bias_mlp,var_bias_mlp = self.meanstditer(miu_bias_old_mlp,var_bias_old_mlp,NS_mlp,instance_number_mlp)
                miu_bias_old_mlp = miu_bias_mlp
                var_bias_old_mlp = var_bias_mlp
                
                if instance_number_mlp <= 1 or update_hyperparameters_bias_mlp == 1:
                    miumin_bias_mlp = miu_bias_mlp
                    stdmin_bias_mlp = std_bias_mlp
                else:
                    if miu_bias_mlp < miumin_bias_mlp:
                        miumin_bias_mlp = miu_bias_mlp

                    if std_bias_mlp < stdmin_bias_mlp:
                        stdmin_bias_mlp = std_bias_mlp

                bias2_mlp_list.append(miu_bias_mlp)
                        

                k_mlp = 1.3 *np.exp(-NS_mlp)+0.7

                # Network variance calculation
                var_mlp = Ez2_mlp - Ez_mlp**2
                NHS_mlp = np.linalg.norm(var_mlp,'fro')

                # Incremental calculation of NHS mean and variance
                miu_var_mlp,std_var_mlp,var_var_mlp = self.meanstditer(miu_var_old_mlp,var_var_old_mlp,NHS_mlp,instance_number_mlp)
                miu_var_old_mlp = miu_var_mlp
                var_var_old_mlp = var_var_mlp

                if instance_number_mlp <= torch.unsqueeze(inputs[-1], 0).size(1) + 1 or update_hyperparameters_var_mlp == 1:
                    miumin_var_mlp = miu_var_mlp
                    stdmin_var_mlp = std_var_mlp
                else:
                    if miu_var_mlp < miumin_var_mlp:
                        miumin_var_mlp = miu_var_mlp

                    if std_var_mlp < stdmin_var_mlp:
                        stdmin_var_mlp = std_var_mlp

                var_mlp_list.append(miu_var_mlp) 

                ro_mlp = 2 * np.exp(-NHS_mlp) + 0.7

                alpha_bias_mlp = optimizer.calculate_alpha_bias(miu_bias_mlp,std_bias_mlp,miumin_bias_mlp,stdmin_bias_mlp,k_mlp)
               
                alpha_var_mlp = optimizer.calculate_alpha_var(miu_var_mlp,std_var_mlp, miumin_var_mlp, stdmin_var_mlp,ro_mlp)
                
                          
                if alpha_bias_mlp >= 1:
                    update_hyperparameters_bias_mlp = 1
                else:
                    update_hyperparameters_bias_mlp = 0      


                if alpha_var_mlp >= 1 and alpha_bias_mlp < 1:
                    update_hyperparameters_var_mlp = 1
                else:
                    update_hyperparameters_var_mlp = 0 
                
                optimizer.updates_hyperparameters(alpha_bias_mlp,alpha_var_mlp,optimizer)
                        
                
            # calculate gradient
            loss.backward()
            # update model weights
            optimizer.step()

            preds = torch.cat(
                    (preds, torch.argmax(mlp.out_act(yhat.detach().reshape(1,self.out.out_features)), axis=1))
            )
        
        return true_labels, preds, mlp
