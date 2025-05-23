import torch
import torchvision
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scienceplots
import argparse as args

torch.manual_seed(0)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_data(N_train, N_test, r, dataset_name="MNIST", download=True, scaling=0.8):
    '''
    Retrieve training and test data
    Args:
        N_train: Number of training data points
        N_test: Number of test data points
        r: desired rank of the data
        dataset_name: ["MNIST", "CIFAR10", "STL"]
        download: True if you want to download the dataset
    Returns:
        SVD of training and testing data
    '''
    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root="./data", train=True, download=download, transform=torchvision.transforms.ToTensor())
        test_dataset = datasets.MNIST(root="./data", train=False, download=download, transform=torchvision.transforms.ToTensor())
        # Shuffle the train and test datasets
        train_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset)))
        test_dataset = torch.utils.data.Subset(test_dataset, torch.randperm(len(test_dataset)))

        train_dataset = torch.utils.data.Subset(train_dataset, range(N_train))
        test_dataset = torch.utils.data.Subset(test_dataset, range(N_test))   
    elif dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=download, transform=torchvision.transforms.ToTensor())
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=download, transform=torchvision.transforms.ToTensor())
        # Shuffle the train and test datasets
        train_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset)))
        test_dataset = torch.utils.data.Subset(test_dataset, torch.randperm(len(test_dataset)))

        train_dataset = torch.utils.data.Subset(train_dataset, range(N_train))
        test_dataset = torch.utils.data.Subset(test_dataset, range(N_test))
    elif dataset_name == "STL":
        T = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
        unsupervised_dataset = datasets.STL10(root="./data", split="unlabeled", download=download,
                                        transform=T)
        # Shuffle the train and test datasets
        whole_dataset = torch.utils.data.Subset(unsupervised_dataset, torch.randperm(N_train + N_test))

        train_dataset = torch.utils.data.Subset(whole_dataset, range(N_train))
        test_dataset = torch.utils.data.Subset(whole_dataset, range(N_train, N_train + N_test))
    else:
        raise ValueError("Wrong Dataset Name")
        
    # Data Preprocessing
    X_train = np.array([np.array(d[0]) for d in train_dataset], dtype=np.float32).reshape(N_train, -1).T
    X_test = np.array([np.array(d[0]) for d in test_dataset], dtype=np.float32).reshape(N_test, -1).T
    X_all = np.concatenate((X_train, X_test), axis=1)
    X_train = X_all[:,:N_train] # (d, N)
    X_test = X_all[:,:N_test]
    
    # Project train and test data to lower dimension using SVD
    U, Sigma, VT = np.linalg.svd(X_train, full_matrices=True)
    U_test, Sigma_test, VT_test = np.linalg.svd(X_test, full_matrices=True)

    # Scale the data: operator norm 
    X_train_projected = U[:, :r] @ np.diag(Sigma[:r]) @ VT[:r, :]
    a_trn = scaling * np.sqrt(N_train) / np.linalg.norm(X_train_projected, ord='fro')
    X_train_projected = U[:, :r] @ (np.diag(Sigma[:r]) * a_trn) @ VT[:r, :]

    X_test_projected = U_test[:, :r] @ np.diag(Sigma_test[:r]) @ VT_test[:r, :]
    a_test = scaling * np.sqrt(N_test) / np.linalg.norm(X_test_projected, ord='fro')
    X_test_projected = U_test[:, :r] @ (np.diag(Sigma_test[:r]) * a_test) @ VT_test[:r, :]

    return U, Sigma[:r] * a_trn, VT, U_test, Sigma_test[:r] * a_test, VT_test


def _get_noskip_theory(d, N_trn, N_val, r, U_trn, Sigma_trn, U_tst, Sigma_tst, 
                       I, eta_trn_squared=0.3, eta_tst_squared=0.3):
    '''
    Get the theoretical values for the model without skip connection.
    '''
    c = d / N_trn
    variance_sum = 0

    j_diagonal_values = []

    L_LT =  U_trn[:, :r].T @ U_tst[:, :r] @ torch.diag(Sigma_tst[:r]) @ torch.diag(Sigma_tst[:r]).T @ U_tst[:, :r].T @ U_trn[:, :r]
    
    for i in np.arange(r):

        if i in I: # if this index is in the selected range
            # for the bias
            j_diagonal_values.append(1 / ((Sigma_trn[i].item() / np.sqrt(eta_trn_squared)) ** 2 + 1))
            # for the variance
            variance_sum += (Sigma_trn[i] ** 2) / (Sigma_trn[i] ** 2 + eta_trn_squared)
        else:
            j_diagonal_values.append(1)

    # use float32
    J = torch.diag(torch.tensor(j_diagonal_values).to(torch.float32).to(device))

    trace_1 = torch.trace(J @ L_LT) 
    trace_2 = (eta_tst_squared) * variance_sum

    bias = trace_1 / N_val
    variance = 1 / d * c / (c - 1) * trace_2

    return bias, variance
    
def _get_skip_theory(d, N_trn, N_val, r, U_trn, Sigma_trn, U_tst, Sigma_tst, 
                     I, eta_trn_squared=0.3, eta_tst_squared=0.3):
    '''
    Get the theoretical values for the model with skip connection.
    '''
    c = d / N_trn
    L_LT = U_trn[:, :r].T @ U_tst[:, :r] @ torch.diag(Sigma_tst[:r]) @ torch.diag(Sigma_tst[:r]).T @ U_tst[:, :r].T @ U_trn[:, :r]
    
    new_k_for_skip = (np.array(I) <= r).sum() # only those indices inside the range of r are counted.
    Sigma_sq = torch.diag(torch.square(Sigma_trn[:r]))
    eta_trn_matrix_squared = torch.diag(torch.tensor([eta_trn_squared] * r).to(device))
    trace_var_1 = torch.trace(Sigma_sq @ torch.linalg.inv(eta_trn_matrix_squared + Sigma_sq))
    trace_var_1 *= eta_tst_squared
    trace_var_1 *= new_k_for_skip * c / ((d ** 2) * (c - 1))
    var_0 = eta_tst_squared
    var_0 *= new_k_for_skip / d

    trace_var_2 = eta_tst_squared* new_k_for_skip / (c * N_trn * d) * torch.trace(torch.linalg.inv(eta_trn_matrix_squared + Sigma_sq) @ (Sigma_sq @ eta_trn_matrix_squared))
    variance = var_0 + trace_var_1 + trace_var_2

    bias = 1 - 2 * new_k_for_skip / d
    bias *= eta_tst_squared
    J_sc = (c * torch.eye(r).to(device) + (c - 1) * Sigma_sq) @ (torch.linalg.inv(c * torch.square(torch.eye(r).to(device) + Sigma_sq @ eta_trn_matrix_squared)))
    # print(J_sc)
    trace_bias_1 = new_k_for_skip / (N_val * d) * torch.trace(J_sc @ L_LT)
    bias += trace_bias_1
    bias += 2 * trace_var_2
    
    return bias, variance

def compare_models_generalization_curves(dataset_name="CIFAR10", r=100, k=50, N_val=4500):
    '''
    FIGURE 1
    Compare the shape of generalization curves for 4 models:
    1. Noisy AE
    2. Reconstruction AE/ PCA
    3. Linear DAE without Skip
    4. Linear DAE with Skip
    '''

    d = 0
    if dataset_name == "MNIST":
        d = 784
    elif dataset_name == "CIFAR10" or dataset_name == "STL":
        d = 3072
    else:
        raise ValueError("Wrong Dataset Name")
        
    scaling=0.8
    eta_trn_squared=0.3
    eta_tst_squared=0.3
    stride_exp = 60
    n_trn_range = np.arange(2568, d - 20, stride_exp) 

    num_trials = 1 
    print(f"Device : {device}")

    # These lists record the validation error for each N_trn.
    val_hist_noisyAE = []
    val_hist_PCA = []
    val_hist_DAE = []
    val_hist_DAE_Skip = []
    
    # For each N_trn, loop over the number of trials and generate new noise. Record the validation loss.
    for i, N_trn in enumerate(tqdm(n_trn_range)):
        val_hist_noisyAE_trials = []
        val_hist_PCA_trials = []
        val_hist_DAE_trials = []
        val_hist_DAE_Skip_trials = []
 
        # Get the signal data for each N_trn & N_val
        U_trn, Sigma_trn, VT_trn, U_tst, Sigma_tst, VT_tst = get_data(N_trn, N_val, r=r, 
                                                                      dataset_name=dataset_name, scaling=scaling) # (d, N)
 
        U_trn = torch.tensor(U_trn).to(torch.float32).to(device)
        Sigma_trn = torch.tensor(Sigma_trn).to(torch.float32).to(device)
        VT_trn = torch.tensor(VT_trn).to(torch.float32).to(device)
        U_tst = torch.tensor(U_tst).to(torch.float32).to(device)
        Sigma_tst = torch.tensor(Sigma_tst).to(torch.float32).to(device)
        VT_tst = torch.tensor(VT_tst).to(torch.float32).to(device)
        X = U_trn[:, :r] @ torch.diag(Sigma_trn[:r] * scaling) @ VT_trn[:r,:]
        X_val = U_tst[:, :r] @ torch.diag(Sigma_tst[:r] * scaling) @ VT_tst[:r, :]

        for trial in range(num_trials):
            # Noise
            A_trn = np.random.randn(X.shape[0], X.shape[1]) / np.sqrt(d) * np.sqrt(eta_trn_squared)
            A_val = np.random.randn(X_val.shape[0], X_val.shape[1]) / np.sqrt(d) * np.sqrt(eta_tst_squared)
            A_trn = torch.tensor(A_trn).to(torch.float32).to(device)
            A_val = torch.tensor(A_val).to(torch.float32).to(device)

            X_noise = X + A_trn
            X_val_noise = X_val + A_val
           
            U, Sigma, Vt = torch.linalg.svd(X_noise, full_matrices=True)

            # 1. Noisy AE : Input X, output X + A
            X_dagger = VT_trn[:r,:].T @ torch.diag(1 / (Sigma_trn[:r] * scaling)) @ U_trn[:, :r].T
            G = X_noise @ VT_trn[:r, :].T @ VT_trn[:r, :] @ X_noise.T
            U_G, Sigma_G, Vt_G = torch.linalg.svd(G, full_matrices=True)
            W1 = U_G[:, :k] @ U_G[:, :k].T @ X_noise @ X_dagger 
            # P_k_X = U[:, :k] @ torch.diag(Sigma[:k]) @ Vt[:k, :]
            # W2 = P_k_X @ X_dagger
            res1 = torch.square(torch.linalg.norm(X_val_noise - W1 @ X_val, ord= "fro")) / X_val.shape[1]

            # 2. Reconstruction : Input X + A, output X + A
            W2 = U[:, :k] @ U[:, :k].T
            res2 = torch.square(torch.linalg.norm(X_val_noise - W2 @ X_val_noise, ord= "fro")) / X_val_noise.shape[1]
            
            # 3. Linear DAE without Skip : Input X + A, output X
            Dr = torch.diag(Sigma[:min(d, N_trn)])
            invdr = torch.tensor(np.linalg.inv(Dr.cpu().numpy())).to(device).to(torch.float32)
            XA_dagger = Vt[:min(d, N_trn), :].T @ invdr @ U[:, :min(d, N_trn)].T
            W3 = U_trn[:, :k] @ torch.diag(Sigma_trn[:k] * scaling) @ VT_trn[:k, :] @ XA_dagger
            res3 = torch.square(torch.linalg.norm(X_val - W3 @ X_val_noise, ord= "fro")) / X_val_noise.shape[1]
            
            # 4. Linear DAE with Skip : Input X + A, output - A
            U_A, Sigma_A, Vt_A = torch.linalg.svd(A_trn, full_matrices=True)
            W4 = - U_A[:, :k] @ torch.diag(Sigma_A[:k]) @ Vt_A[:k, :] @ XA_dagger
            res4 = torch.square(torch.linalg.norm(-A_val - W4 @ X_val_noise, ord= "fro")) / X_val_noise.shape[1]
            
            val_hist_noisyAE_trials.append(res1)
            val_hist_PCA_trials.append(res2)
            val_hist_DAE_trials.append(res3)
            val_hist_DAE_Skip_trials.append(res4)
            
        val_hist_noisyAE.append(torch.mean(torch.tensor(val_hist_noisyAE_trials)))
        val_hist_PCA.append(torch.mean(torch.tensor(val_hist_PCA_trials)))
        val_hist_DAE.append(torch.mean(torch.tensor(val_hist_DAE_trials)))
        val_hist_DAE_Skip.append(torch.mean(torch.tensor(val_hist_DAE_Skip_trials)))

        
    colors = ["navy",  "orange" , "red", "blue"]
    c_trn_empirical = [d / n_trn for n_trn in n_trn_range]
    plt.style.use('science')

    plt.plot(c_trn_empirical, val_hist_noisyAE, color=colors[0], linestyle='--', linewidth=2, label="Noisy AE")
    plt.plot(c_trn_empirical, val_hist_PCA, color=colors[1], linestyle='--', linewidth=2, label="AE/PCA")
    plt.plot(c_trn_empirical, val_hist_DAE, color=colors[2], linestyle='--', linewidth=2, label="DAE")
    plt.plot(c_trn_empirical, val_hist_DAE_Skip, color=colors[3], linestyle='--', linewidth=2, label="DAE+skip")

    plt.plot(c_trn_empirical, val_hist_noisyAE, color=colors[0], linestyle='None', marker=".", markersize=17)
    plt.plot(c_trn_empirical, val_hist_PCA, color=colors[1], linestyle='None', marker=".", markersize=17)
    plt.plot(c_trn_empirical, val_hist_DAE, color=colors[2], linestyle='None', marker=".", markersize=17)
    plt.plot(c_trn_empirical, val_hist_DAE_Skip, color=colors[3], linestyle='None', marker=".", markersize=17)

 
    plt.axvline(1, linewidth=1, color="black", alpha=0.5, ls= "-")
    plt.xlabel(r"$c = \frac{d}{n}$", size=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xticks([1.0, 1.05, 1.10, 1.15, 1.20])
    plt.ylabel("Test Error", size=32)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()


def compare_different_bottleneck_size(dataset_name, r, N_val=4500):
    '''
        Figure 2
        bottleneck_sizes: A list of positive integers representing bottleneck dimensions, where each value satisfies 1 ≤ k ≤ r.
        Is: A list of critical point specifications corresponding to each bottleneck size. 
            Each element in Is is a list that defines a critical point.
            For example, a global minimizer would be represented as: [1, ... ,k]
    '''    
    if dataset_name == "MNIST":
        d = 784
    elif dataset_name == "CIFAR10" or dataset_name == "STL":
        d = 3072
    else:
        raise ValueError("Wrong dataset name")
        
    scaling=0.8
    eta_trn_squared=0.3
    eta_tst_squared=0.3
    stride_theory = 20
    n_trn_theory_range = np.arange(2048, d - 20, stride_theory)
    # n_trn_theory_range = np.arange(2048, d - 20, 500)
    bottleneck_sizes = [1, 10, 30, 50, 100]
    Is = [np.arange(i) for i in bottleneck_sizes]
        
#     device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device : {device}")
    
    # Variance Theory
    theory_var_noskip_ks = []
    theory_var_skip_ks = []

    # Bias Theory
    theory_bias_noskip_ks = []
    theory_bias_skip_ks = []

    # Validation loss theory
    theory_val_noskip_ks = []
    theory_val_skip_ks = []
    
    for i_k, k in enumerate(bottleneck_sizes):
        # Variance Theory
        theory_var_noskip = []
        theory_var_skip = []

        # Bias Theory
        theory_bias_noskip = []
        theory_bias_skip = []

        # Validation loss theory
        theory_val_noskip = []
        theory_val_skip = []
        
        for N_trn in tqdm(n_trn_theory_range):

            if N_trn > d: break
            U_trn, Sigma_trn, _, U_tst, Sigma_tst, _ = get_data(N_trn, N_val, r=r, dataset_name=dataset_name, scaling=scaling)

            U_trn = torch.tensor(U_trn).to(device).to(torch.float32)
            Sigma_trn = torch.tensor(Sigma_trn).to(device).to(torch.float32) * scaling
            U_tst = torch.tensor(U_tst).to(device).to(torch.float32)
            Sigma_tst = torch.tensor(Sigma_tst).to(device).to(torch.float32) * scaling

            ## No Skip
            bias, variance = _get_noskip_theory(d, N_trn, N_val, r, U_trn, Sigma_trn, U_tst, Sigma_tst, 
                                                Is[i_k], eta_trn_squared=eta_trn_squared, 
                                                eta_tst_squared=eta_tst_squared)
            
            theory = bias + variance
            theory_bias_noskip.append(bias.item())
            theory_var_noskip.append(variance.item())
            theory_val_noskip.append(theory.item())
            
            ## Skip
            bias, variance = _get_skip_theory(d, N_trn, N_val, r, U_trn, Sigma_trn, U_tst, Sigma_tst, 
                                              Is[i_k], eta_trn_squared=eta_trn_squared, 
                                              eta_tst_squared=eta_tst_squared)
            
            theory = bias + variance

            theory_val_skip.append(theory.item())
            theory_var_skip.append(variance.item())
            theory_bias_skip.append(bias.item())      
            
        theory_val_noskip_ks.append(theory_val_noskip)
        theory_val_skip_ks.append(theory_val_skip)
        
        theory_var_noskip_ks.append(theory_var_noskip)
        theory_var_skip_ks.append(theory_var_skip)
        
        theory_bias_noskip_ks.append(theory_bias_noskip)
        theory_bias_skip_ks.append(theory_bias_skip)
        
    # Plot
    colors_for_noskip = ["midnightblue", "mediumblue", "mediumpurple", "violet", "deeppink"]
    colors_for_skip = ["red", "orange", "green", "rosybrown", "silver"]


    c_trn_theory = [d / n_trn for n_trn in n_trn_theory_range]

    plt.style.use('science')

    for i, bottleneck_k in enumerate(bottleneck_sizes):
        plt.plot(c_trn_theory, theory_val_noskip_ks[i], label=f"k = {bottleneck_k}", linestyle='-', color=colors_for_noskip[i], linewidth=1.8)
    
    plt.xlabel("c", size=32)
    # plt.ylim(top=0.9)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    # plt.xticks([1.0, 1.05, 1.10, 1.15, 1.20])
    plt.ylabel("Test Error", size=32)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()

    # Test Loss for skip
    for i, bottleneck_k in enumerate(bottleneck_sizes):
        plt.plot(c_trn_theory, theory_val_skip_ks[i], label=f"k = {bottleneck_k}", linestyle='-', color=colors_for_skip[i], linewidth=1.8)
    plt.xlabel("c", size=32)
    # plt.ylim(top=0.43)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    # plt.xticks([1.0, 1.05, 1.10, 1.15, 1.20])
#     plt.yticks([0.39, 0.41, 0.43])
    plt.ylabel("Test Error", size=32)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()

    # variance
    for i, bottleneck_k in enumerate(bottleneck_sizes):
        plt.plot(c_trn_theory, theory_var_noskip_ks[i], label=f"k = {bottleneck_k}", linestyle='-', color=colors_for_noskip[i], linewidth=1.8)
        plt.xlabel("c", size=32)
    # plt.ylim(top=0.9)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylabel("Variance", size=32)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()

    for i, bottleneck_k in enumerate(bottleneck_sizes):
        plt.plot(c_trn_theory, theory_var_skip_ks[i], label=f"k = {bottleneck_k}", linestyle='-', color=colors_for_skip[i], linewidth=1.8)
    plt.xlabel("c", size=32)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylabel("Variance", size=32)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()

    # bias
    for i, bottleneck_k in enumerate(bottleneck_sizes):
        # Theoretical Values
        plt.plot(c_trn_theory, theory_bias_noskip_ks[i], label=f"k = {bottleneck_k}", linestyle='-', color=colors_for_noskip[i], linewidth=1.8)
    plt.xlabel(r"$c = \frac{d}{n}$", size=32)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylabel("Bias", size=32)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()
    for i, bottleneck_k in enumerate(bottleneck_sizes):
        # Theoretical Values
        plt.plot(c_trn_theory, theory_bias_skip_ks[i], label=f"k = {bottleneck_k}", linestyle='-', color=colors_for_skip[i], linewidth=1.8)
    plt.xlabel(r"$c = \frac{d}{n}$", size=32)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylabel("Bias", size=32)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()

def compare_skip_noskip(dataset_name, r, k, N_val=4500):
    '''
    Figure 3
    
    '''
    if dataset_name == "MNIST":
        d = 784
    elif dataset_name == "CIFAR10" or dataset_name == "STL":
        d = 3072
    else:
        raise ValueError("Wrong dataset name")
        
    scaling=0.8
    eta_trn_squared=0.3
    eta_tst_squared=0.3
    stride_theory=20 

    n_trn_range_theory = np.arange(2568, d - 20, stride_theory) # for theory
    
    # For empirical values
    n_trn_range = np.arange(2568, d - 20, stride_theory * 3)
    n_trn_range_underparameterized = np.arange(d + 20, 3512, stride_theory * 3)
#     n_trn_range = np.concatenate((n_trn_range, n_trn_range_underparameterized))
    
    theory_bias_noskip = []
    theory_bias_skip = []
    theory_var_noskip = []
    theory_var_skip = []
    theory_val_noskip = []
    theory_val_skip = []
    
    # Empirical Values
    val_hist_skip_mean = []
    val_hist_noskip_mean = []

    var_hist_skip_mean = []
    var_hist_noskip_mean = []

    bias_hist_skip_mean = []
    bias_hist_noskip_mean = []
    
    under_val_hist_skip_mean = []
    under_val_hist_noskip_mean = []

    under_var_hist_skip_mean = []
    under_var_hist_noskip_mean = []

    under_bias_hist_skip_mean = []
    under_bias_hist_noskip_mean = []
    
    # theoretical values
    print("Theory + overparameterized")
    for N_trn in tqdm(n_trn_range_theory):
        # data
        U_trn, Sigma_trn, Vt_trn, U_tst, Sigma_tst, Vt_tst = get_data(N_trn, N_val, r=r, dataset_name=dataset_name, scaling=scaling)

        U_trn = torch.tensor(U_trn).to(device).to(torch.float32)
        Vt_trn = torch.tensor(Vt_trn).to(device).to(torch.float32)
        Sigma_trn = torch.tensor(Sigma_trn).to(device).to(torch.float32) * scaling
        U_tst = torch.tensor(U_tst).to(device).to(torch.float32)
        Vt_tst = torch.tensor(Vt_tst).to(device).to(torch.float32)
        Sigma_tst = torch.tensor(Sigma_tst).to(device).to(torch.float32) * scaling
        X = U_trn[:, :r] @ torch.diag(Sigma_trn[:r]) @ Vt_trn[:r, :]
        X_val = U_tst[:, :r] @ torch.diag(Sigma_tst[:r]) @ Vt_tst[:r, :]
          
        ## No Skip
        bias, variance = _get_noskip_theory(d, N_trn, N_val, r, U_trn, Sigma_trn, U_tst, Sigma_tst, 
                                            I=np.arange(k), eta_trn_squared=eta_trn_squared, 
                                            eta_tst_squared=eta_tst_squared)

        theory = bias + variance
        theory_bias_noskip.append(bias.item())
        theory_var_noskip.append(variance.item())
        theory_val_noskip.append(theory.item())

        ## Skip
        bias, variance = _get_skip_theory(d, N_trn, N_val, r, U_trn, Sigma_trn, U_tst, Sigma_tst, 
                                          I=np.arange(k), eta_trn_squared=eta_trn_squared, 
                                          eta_tst_squared=eta_tst_squared)

        theory = bias + variance

        theory_val_skip.append(theory.item())
        theory_var_skip.append(variance.item())
        theory_bias_skip.append(bias.item())      
        
        # Empirical Values
        val_hist_skip_mean = []
        val_hist_noskip_mean = []

        var_hist_skip_mean = []
        var_hist_noskip_mean = []

        bias_hist_skip_mean = []
        bias_hist_noskip_mean = []

        n_trials = 1

        n_trn_range_emp = []
        n_trn_range_emp_under = []
        
        if N_trn in n_trn_range:
            val_hist_skip_Ntrn = []
            val_hist_noskip_Ntrn = []
            var_hist_skip_Ntrn = []
            var_hist_noskip_Ntrn = []
            bias_hist_skip_Ntrn = []
            bias_hist_noskip_Ntrn = []

            for trial in range(n_trials):
                # New noise
                A_trn = np.random.randn(X.shape[0], X.shape[1]) / np.sqrt(d) * np.sqrt(eta_trn_squared)
                A_val = np.random.randn(X_val.shape[0], X_val.shape[1]) / np.sqrt(d) * np.sqrt(eta_tst_squared)
                print(f"SNR : {Sigma_trn[:r].max() / np.linalg.norm(A_trn, ord=2)}") # 'fro' for the default
                print(f"SNR test : {Sigma_tst[:r].max() / np.linalg.norm(A_val, ord=2)}") # 'fro' for the default
                A_trn = torch.tensor(A_trn).to(torch.float32).to(device)
                A_val = torch.tensor(A_val).to(torch.float32).to(device)

                X_noise = X + A_trn
                X_val_noise = X_val + A_val
                # 1. X + A = U @ D @ Vt
                U, Sigma, Vt = torch.linalg.svd(X_noise, full_matrices=True)

                # 2. Extract the first r rows of Ut @ X and denote it as Wr r is the rank of X + A.
                Wr_skip = - A_trn @ Vt.T # skip connection
                Wr = X @ Vt.T
                # 3. get W
                Uwr_skip, Sigmawr_skip, Vwrt_skip = torch.linalg.svd(Wr_skip, full_matrices=True)
                Pkwr_skip = Uwr_skip[:, :k] @ torch.diag(Sigmawr_skip[:k]) @ Vwrt_skip[:k, :]

                Dr = torch.diag(Sigma[:min(d, N_trn)])

                Uwr, Sigmawr, Vwrt = torch.linalg.svd(Wr, full_matrices=True)
                Pkwr = Uwr[:, :k] @ torch.diag(Sigmawr[:k]) @ Vwrt[:k, :] if k < r else Uwr[:, :r] @ torch.diag(Sigmawr[:r]) @ Vwrt[:r, :]

                invdr = torch.tensor(np.linalg.inv(Dr.cpu().numpy())).to(device).to(torch.float32)

                W_skip = Pkwr_skip @ torch.concatenate((invdr, torch.zeros((N_trn, d - N_trn)).to(device)), axis=1) @ U.T
                W = Pkwr @ torch.concatenate((invdr, torch.zeros((N_trn, d - N_trn)).to(device)), axis=1) @ U.T


                # get the generalization loss.
                res_skip = torch.square(torch.linalg.norm((-A_val - W_skip @ X_val_noise), ord="fro")) / X_val.shape[1] # skip connection
                val_hist_skip_Ntrn.append(res_skip)
                print(f"[SKIP] TRIAL {trial + 1}/{n_trials} Val loss of dim {d} and N {N_trn} : {res_skip}")

                res = torch.square(torch.linalg.norm(X_val - W @ X_val_noise, ord= "fro")) / X_val.shape[1]
                val_hist_noskip_Ntrn.append(res)
                print(f"[NO SKIP] TRIAL {trial + 1}/{n_trials}Val loss of dim {d} and N {N_trn} : {res}")

                var_skip = torch.square(torch.linalg.norm(W_skip, ord='fro')) * eta_tst_squared / d
                var = torch.square(torch.linalg.norm(W, ord='fro')) * eta_tst_squared / d
                print(f"[SKIP] TRIAL {trial + 1}/{n_trials} Var {var_skip}")
                print(f"[NO SKIP]TRIAL {trial + 1}/{n_trials} Var {var}")

                var_hist_skip_Ntrn.append(var_skip)
                var_hist_noskip_Ntrn.append(var)

                bias_noskip = res - var
                bias_skip = res_skip - var_skip
                bias_hist_skip_Ntrn.append(bias_skip)
                bias_hist_noskip_Ntrn.append(bias_noskip)

            val_hist_skip_mean.append(torch.mean(torch.tensor(val_hist_skip_Ntrn)))
            val_hist_noskip_mean.append(torch.mean(torch.tensor(val_hist_noskip_Ntrn)))

            var_hist_skip_mean.append(torch.mean(torch.tensor(var_hist_skip_Ntrn)))
            var_hist_noskip_mean.append(torch.mean(torch.tensor(var_hist_noskip_Ntrn)))

            bias_hist_skip_mean.append(torch.mean(torch.tensor(bias_hist_skip_Ntrn)))
            bias_hist_noskip_mean.append(torch.mean(torch.tensor(bias_hist_noskip_Ntrn)))
            n_trn_range_emp.append(N_trn)
    print("underparameterized")
    for N_trn in tqdm(n_trn_range_underparameterized):
        val_hist_skip_Ntrn = []
        val_hist_noskip_Ntrn = []
        var_hist_skip_Ntrn = []
        var_hist_noskip_Ntrn = []
        bias_hist_skip_Ntrn = []
        bias_hist_noskip_Ntrn = []

        U_trn, Sigma_trn, Vt_trn, U_tst, Sigma_tst, Vt_tst = get_data(N_trn, N_val, r=r, dataset_name=dataset_name, scaling=scaling)

        U_trn = torch.tensor(U_trn).to(device).to(torch.float32)
        Vt_trn = torch.tensor(Vt_trn).to(device).to(torch.float32)
        Sigma_trn = torch.tensor(Sigma_trn).to(device).to(torch.float32) * scaling
        U_tst = torch.tensor(U_tst).to(device).to(torch.float32)
        Vt_tst = torch.tensor(Vt_tst).to(device).to(torch.float32)
        Sigma_tst = torch.tensor(Sigma_tst).to(device).to(torch.float32) * scaling
        X = U_trn[:, :r] @ torch.diag(Sigma_trn[:r]) @ Vt_trn[:r, :]
        X_val = U_tst[:, :r] @ torch.diag(Sigma_tst[:r]) @ Vt_tst[:r, :]

        for trial in range(n_trials):
            # New noise
            A_trn = np.random.randn(X.shape[0], X.shape[1]) / np.sqrt(d) * np.sqrt(eta_trn_squared)
            A_val = np.random.randn(X_val.shape[0], X_val.shape[1]) / np.sqrt(d) * np.sqrt(eta_tst_squared)
            print(f"SNR : {Sigma_trn[:r].max() / np.linalg.norm(A_trn, ord=2)}") # 'fro' for the default
            print(f"SNR test : {Sigma_tst[:r].max() / np.linalg.norm(A_val, ord=2)}") # 'fro' for the default
            A_trn = torch.tensor(A_trn).to(torch.float32).to(device)
            A_val = torch.tensor(A_val).to(torch.float32).to(device)

            X_noise = X + A_trn
            X_val_noise = X_val + A_val
            # 1. X + A = U @ D @ Vt
            U, Sigma, Vt = torch.linalg.svd(X_noise, full_matrices=True)

            # 2. Extract the first r rows of Ut @ X and denote it as Wr r is the rank of X + A.
            Wr_skip = - A_trn @ (Vt.T[:, :X.shape[0]])

            Wr = X @ (Vt.T[:, :X.shape[0]])
            # 3. get W
            Uwr_skip, Sigmawr_skip, Vwrt_skip = torch.linalg.svd(Wr_skip, full_matrices=True)
            Pkwr_skip = Uwr_skip[:, :k] @ torch.diag(Sigmawr_skip[:k]) @ Vwrt_skip[:k, :]

            Dr = torch.diag(Sigma[:min(d, N_trn)])

            Uwr, Sigmawr, Vwrt = torch.linalg.svd(Wr, full_matrices=True)
            Pkwr = Uwr[:, :k] @ torch.diag(Sigmawr[:k]) @ Vwrt[:k, :] if k < r else Uwr[:, :r] @ torch.diag(Sigmawr[:r]) @ Vwrt[:r, :]

            invdr = torch.tensor(np.linalg.inv(Dr.cpu().numpy())).to(device).to(torch.float32)
            # print(X.shape)
            # print(Pkwr_skip.shape)
            # print(invdr.shape)
            W_skip = Pkwr_skip @ invdr @ U.T
            W = Pkwr @ invdr @ U.T

            # get the generalization loss.
            res_skip = torch.square(torch.linalg.norm((-A_val - W_skip @ X_val_noise), ord="fro")) / X_val.shape[1] # skip connection
            print(f"[SKIP] TRIAL {trial + 1}/{n_trials} Val loss of dim {d} and N {N_trn} : {res_skip}")

            res = torch.square(torch.linalg.norm(X_val - W @ X_val_noise, ord= "fro")) / X_val.shape[1]
            print(f"[NO SKIP] TRIAL {trial + 1}/{n_trials}Val loss of dim {d} and N {N_trn} : {res}")

            var_skip = torch.square(torch.linalg.norm(W_skip, ord='fro')) * eta_tst_squared / d
            var = torch.square(torch.linalg.norm(W, ord='fro')) * eta_tst_squared / d
            print(f"[SKIP] TRIAL {trial + 1}/{n_trials} Var {var_skip}")
            print(f"[NO SKIP]TRIAL {trial + 1}/{n_trials} Var {var}")

            val_hist_skip_Ntrn.append(res_skip)
            val_hist_noskip_Ntrn.append(res)
            var_hist_skip_Ntrn.append(var_skip)
            var_hist_noskip_Ntrn.append(var)

            bias_noskip = res - var
            bias_skip = res_skip - var_skip
            bias_hist_skip_Ntrn.append(bias_skip)
            bias_hist_noskip_Ntrn.append(bias_noskip)

        under_val_hist_skip_mean.append(torch.mean(torch.tensor(val_hist_skip_Ntrn)))
        under_val_hist_noskip_mean.append(torch.mean(torch.tensor(val_hist_noskip_Ntrn)))

        under_var_hist_skip_mean.append(torch.mean(torch.tensor(var_hist_skip_Ntrn)))
        under_var_hist_noskip_mean.append(torch.mean(torch.tensor(var_hist_noskip_Ntrn)))

        under_bias_hist_skip_mean.append(torch.mean(torch.tensor(bias_hist_skip_Ntrn)))
        under_bias_hist_noskip_mean.append(torch.mean(torch.tensor(bias_hist_noskip_Ntrn)))
        n_trn_range_emp_under.append(N_trn)

        
    # colors_for_noskip = ["blue", "navy", "magenta"]
    # # reddish colors for skip
    # colors_for_skip = ["red", "orange", "green"]

    # first, with no skip connection.
    c_trn_empirical = [d / n_trn for n_trn in n_trn_range_emp]
    c_trn_empirical_under = [d / n_trn for n_trn in n_trn_range_emp_under]
    # c_trn_empirical = [d / n_trn for n_trn in n_trn_range]
    # c_trn_empirical_under = [d / n_trn for n_trn in n_trn_range_underparameterized]
    # c_trn = np.concatenate((c_trn_empirical_under, c_trn_empirical))

    c_trn_theory = [d / n_trn for n_trn in n_trn_range_theory]

    plt.style.use('science')
    
    # Empirical values no skip
    plt.plot(c_trn_empirical, val_hist_noskip_mean, color="blue", linestyle='None', marker=".", markersize=17)
    plt.plot(c_trn_empirical_under, under_val_hist_skip_mean, color="blue", ls='--')
    # Theoretical values
    plt.plot(c_trn_theory, theory_val_noskip, label=f"k = {k}", color="blue", ls="-")

    # Empirical values skip
    plt.plot(c_trn_empirical, val_hist_skip_mean, color="red", marker=".", linestyle='None', markersize=17)
    plt.plot(c_trn_empirical_under, under_val_hist_skip_mean, color="red", ls='--')
    # Theoretical values
    plt.plot(c_trn_theory, theory_val_skip, label=f"k = {k}, +sc", color="red", ls="-")
    
    plt.axvline(1, linewidth=1, color="black", alpha=0.5, ls= "-")
    plt.xlabel(r"$c = \frac{d}{n}$", size=32)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel("Test Error", size=32)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()

    # variance of no skip
    plt.plot(c_trn_empirical, var_hist_noskip_mean, color="blue", linestyle='None', marker=".", markersize=17)
    plt.plot(c_trn_empirical_under, under_var_hist_noskip_mean, color="blue", ls='--')
    # Theoretical values
    plt.plot(c_trn_theory, theory_var_noskip, label=f"k = {k}", color="blue", ls="-")

    plt.plot(c_trn_empirical, var_hist_skip_mean, color="red", marker=".", linestyle='None', markersize=17)
    plt.plot(c_trn_empirical_under, under_var_hist_skip_mean, color="red", ls='--')
    # Theoretical values
    plt.plot(c_trn_theory, theory_var_skip, label=f"k = {k}, +sc", color="red", ls="-")
    
    plt.axvline(1, linewidth=1, color="black", alpha=0.5, ls= "-")
    plt.xlabel(r"$c = \frac{d}{n}$", size=32)
    plt.ylabel("Variance", size=32)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()

    # Empirical values
    plt.plot(c_trn_empirical, bias_hist_noskip_mean, color="blue", linestyle='None', marker=".", markersize=17)
    plt.plot(c_trn_empirical_under, under_bias_hist_noskip_mean, color="blue", ls='--')
    # Theoretical values
    # axs[0].plot(n_trn_theory_range, th_valloss_noskip_ks[i], label=f"Theoretical, I^x = [{start_k - 1, right_end}]", ls="-")
    plt.plot(c_trn_theory, theory_bias_noskip, label=f"k = {k}", color="blue", ls="-")

    plt.plot(c_trn_empirical, bias_hist_skip_mean, color="red", marker=".", linestyle='None', markersize=17)
    plt.plot(c_trn_empirical_under, under_bias_hist_skip_mean, color="red", linestyle='--')
    # Theoretical values
    plt.plot(c_trn_theory, theory_bias_skip, label=f"k = {k}, +sc", color="red", ls="-")
    
    plt.axvline(1, linewidth=1, color="black", alpha=0.5, ls= "-")
    plt.xlabel(r"$c = \frac{d}{n}$", size=32)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel("Bias", size=32)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()
            

if __name__ == "__main__":
    parser = args.ArgumentParser()
    parser.add_argument('--figure', '-f', type=int, required=True, help="specify the number of figure in the paper") 
    parser.add_argument('--rank', '-r', type=int, default=100, help="specify the rank of the data") 
    parser.add_argument('--bottleneck', '-b', type=int, default=50, help="specify the bottleneck dimension") 
    parser.add_argument('--dataset', '-d', default="CIFAR10", choices=['MNIST', 'CIFAR10', 'STL'], help="specify the dataset") 
    args = parser.parse_args()
    if args.figure == 1:
        compare_models_generalization_curves(dataset_name=args.dataset, r=args.rank)
    elif args.figure == 2:
        compare_different_bottleneck_size(dataset_name=args.dataset, r=args.rank)
    else:
        compare_skip_noskip(dataset_name=args.dataset, r=args.rank, k=args.bottleneck)