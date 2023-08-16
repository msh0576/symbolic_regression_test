from scipy.signal import savgol_filter
import numpy as np



def preprocess(df, drop_labels, window_length, poly_order):
    '''
    Return:
        X_train_scale:
        Xdot: 
        y_train: tensor, 
    '''
    # drop some features (unit_number, time_cycles, RUL)
    X_train=df.drop(columns=drop_labels).copy()
    y_train = df['RUL'].copy()
    
    
    # scaling
    X_train_scale = scaler.fit_transform(X_train)
    
    
    # calculate Xdot
    t_max = X_train.shape[0]
    t = generate_list_increasing_by_dt(size=t_max, dt=1)
    differentiation_method = ps.SmoothedFiniteDifference(smoother=savgol_filter, 
                                smoother_kws={'window_length': window_length, 'polyorder': poly_order})
    
    Xdot = differentiation_method._differentiate(X_train_scale, t)
    
    return X_train_scale, Xdot, y_train.values


def generate_list_increasing_by_dt(size, dt=0.1, current_value=0):
    output_list = []

    for _ in range(size):
        output_list.append(current_value)
        current_value += dt

    return np.array(output_list)