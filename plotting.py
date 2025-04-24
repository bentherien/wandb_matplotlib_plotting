import math
import pprint
import re
import matplotlib.colors as mcolors
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from scipy.signal import savgol_filter

def exponential_moving_average(data, alpha=0.1):
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return ema

def rolling_average(data, 
                    window_size=4, 
                    sigma=2.0, 
                    alpha=0.1, 
                    polyorder=4, 
                    deriv=0,
                    delta=1.0,
                    mode='interp',
                    tpe='savgol'):
    """Compute a rolling average over a 1D array."""
    if tpe =='rolling':
        cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
        ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
        return ma_vec
    elif tpe == 'ema':
        return exponential_moving_average(data, alpha)
    elif tpe == 'savgol':
        return savgol_filter(data, 
                            window_length=window_size, 
                            polyorder=polyorder,
                            deriv=deriv, 
                            delta=delta, 
                            axis=-1, 
                            mode='interp', 
                            cval=0.0)
    elif tpe == 'super':
        data = gaussian_filter1d(data,sigma=sigma)
        cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
        ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
        return ma_vec
    elif tpe == 'gaussian':
        return gaussian_filter1d(data,sigma=sigma)
    elif tpe == 'none':
        return data
    else:
        raise NotImplementedException()
    

def c_plot(values,
           data,
           title=None,
           smooth_mean=True,
           smooth_std=True,
           ylim=None,
           xlim=None,
           xlab="Training Iterations",
           ylab="Train Loss",
           savepath=None,
           figsize=(6, 4,),
           legend_fs=18,
           ylab_fs=15,
           xlab_fs=15,
           lab_suffix=None,
           log_cmap=False,
           use_std=True,
           linestyle_ovrr=dict(),
           verbose=False,
           order=order,
           use_legend=True,
           legend_loc='best',
           ovr_legend=None,
           use_colormap=False,
           color_value_map={},
           mapper_key='width',
           reorder=True,
           ylog=False,
           colormap=plt.cm.plasma,
           hline=[],
           vline=[],
           label_dict=[],
           skipfactor=10,
           sciy=False,
           linewidth=1,
           plot_with_steps=False,
           remove_border=True,
           autocenter=False,
           threshold_std=False,
           bottom_zero=False,
           filter_args=dict(tpe='savgol',window_size=2, deriv=0.0, delta=1.0, polyorder=1,),
           show_meta_train_boundary=True):
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    if ylog:
        plt.yscale('log')
        


    if use_colormap:
        numerical_values = [color_value_map[k] for k in values]
        if log_cmap:
            norm = mcolors.LogNorm(vmin=min(numerical_values)+1e-4, vmax=max(numerical_values))
        else:
            norm = mcolors.Normalize(vmin=min(numerical_values), vmax=max(numerical_values))
        # Create a ScalarMappable object with the chosen colormap and normalization
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    else:
        mapper = None

    eymin, eymax = [],[]
    for i, (y, df) in enumerate(data.items()):
        if verbose:
            print(f"Plotting element {i}, with key: {y}")

        #########################################################
        # setup marker, color, label, linestyle per curve
        #########################################################
        m,c,lab,ls = ovr_legend[y]
        if lab_suffix is not None:
            lab = lab + lab_suffix[i]
        if mapper:
            c = mapper.to_rgba(idx)
        

        # Don't smooth if the first 20 values are greater than 20
        # since these curves likely diverge
        no_smooth = len(np.where(df['mean'][:20] >= 20)[0]) > 0

        # Set the values to 20 if they are greater than 20
        # since these curves likely diverge and we want 
        # them to be visible when they are smoothed
        df['mean'][np.where(df['mean'] >= 20)[0]] = 20
        df['stderr'][np.where(df['stderr'] >= 20)[0]] = 20

        yval = df['mean'][::skipfactor].astype(np.float32)
        steps = df['steps'][::skipfactor].astype(np.float32)
        stderr = df['stderr'][::skipfactor].astype(np.float32)

        max_vals = yval + stderr
        min_vals = yval - stderr

        #########################################################
        # Set the x limits
        #########################################################
        if xlim is not None:
            a = np.where(steps > xlim[-1])[0]
            if len(a) != 0:
                steps = steps[:a[0]]
                yval = yval[:a[0]]
                max_vals = max_vals[:a[0]]
                min_vals = min_vals[:a[0]]

        #########################################################
        # Smooth the curve
        #########################################################
        yval_before = yval.shape[0]
        if smooth_mean and no_smooth == False:
            yval = rolling_average(yval, **filter_args)
            if yval_before != yval.shape[0]:
                steps = steps[:yval.shape[0]]
                stderr = stderr[:yval.shape[0]]
                max_vals = max_vals[:yval.shape[0]]
                min_vals = min_vals[:yval.shape[0]]
                

        # save max and min values attained
        eymin.append(np.min(yval))
        eymax.append(yval[0])

        if verbose:
            print("plotting ",lab)


        #########################################################
        # Plotting the mean curve
        #########################################################
        plt.plot(steps, yval, label=lab, color=c, linestyle=linestyle_ovrr.get(i, ls), linewidth=linewidth)

        
        #########################################################
        # Plotting the std bars
        #########################################################
        if use_std:
            if smooth_std:
                assert max_vals.shape == yval.shape, "max_vals.shape {} != yval.shape {}".format(max_vals.shape, yval.shape)
                assert min_vals.shape == yval.shape, "min_vals.shape {} != yval.shape {}".format(min_vals.shape, yval.shape)
                
                max_vals = rolling_average(max_vals,**filter_args) 
                min_vals = rolling_average(min_vals,**filter_args)

                assert steps.shape == max_vals.shape, "max_vals.shape {} != steps.shape {}".format(max_vals.shape, steps.shape)

                #threshold STD bars for readability
                if threshold_std:
                    diff = 3
                    noisemax = np.where(max_vals - diff > yval)
                    max_vals[noisemax] = yval[noisemax] + diff
                    
                    noisemin = np.where(min_vals + diff < yval)
                    min_vals[noisemin] = yval[noisemin] - diff

                    
                plt.fill_between(steps, 
                                  max_vals,
                                  min_vals,         
                                  color=c, 
                                  alpha=0.2)
            else:
                plt.fill_between(steps, 
                                 max_vals, 
                                 min_vals, 
                                 color=c, 
                                 alpha=0.2)
        
        #########################################################
        # END FOR
        #########################################################                         
                
            
    if sciy:
        # Set the formatter for the y-axis of this specific axis
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))  # Use scientific notation if outside this range

        ax.yaxis.set_major_formatter(formatter)

        # Optionally, add an offset text (scientific notation) to the axis
        ax.yaxis.get_offset_text().set_visible(True)
    
    if title is not None:
        plt.title(title)
    
    for x in vline:
        ax.axvline(**x)
        
    for x in hline:
        ax.axhline(**x)
    
    if type(ylim) == dict:
        plt.ylim(**ylim)
    else:
        plt.ylim(ylim)

    plt.grid(True, linestyle='--', )
    plt.xlabel(xlab,fontsize=xlab_fs)
    plt.ylabel(ylab,fontsize=ylab_fs)
    
    
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    # Manually set the color of the tick labels

    # Set the color of the spines
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.spines['top'].set_color('gray')   # If you decide to keep the top spine
    ax.spines['right'].set_color('gray') 
    
    if remove_border:
        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Optionally customize the remaining spines and ticks
        ax.spines['left'].set_position(('outward', 0))
        ax.spines['bottom'].set_position(('outward', 0))
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
    for tick in ax.get_xticklabels():
        tick.set_color('black')
        
    for tick in ax.get_yticklabels():
        tick.set_color('black')
    

    if show_meta_train_boundary:
        ax.axvline(x=1000, color='darkred', linestyle='--',linewidth=1.5) #,label="Meta-train Horizon")
        
    if use_legend:
        leg = plt.legend(fontsize=legend_fs,loc=legend_loc)
            
        # Adjust line width for each line in the legend
        for handle in leg.legend_handles:
            handle.set_linewidth(3.0)


    #########################################################
    # Automatic y-axis limits with outlier rejection
    #########################################################
    if autocenter:
        def reject_outliers(data, m=2):
            if len(data) == 1:
                return data
            try:
                out = data[abs(data - np.mean(data)) < m * np.std(data)]
            except Exception:
                print("Exception reject_outliers data", data)
                raise Exception
            return out

        if verbose:
            print("eymin",eymin)
            print("eymax",eymax)

        eymin = [x for x in eymin if str(x) != 'nan']
        eymax = [x for x in eymax if str(x) != 'nan']

        if verbose:
            print("eymin",eymin)
            print("eymax",eymax)

        
        eymax = np.max(reject_outliers(np.array(eymax)))
        eymin = np.array(eymin)
        eymin = np.min(eymin[eymin >= 0])

        #margin is 5% of graph
        margin = (eymax - eymin)  * 0.05

        if bottom_zero:
            plt.ylim(0,eymax+margin)
        else:
            plt.ylim(eymin-margin,eymax+margin)
    
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath,
                    bbox_inches='tight')
    plt.show()
