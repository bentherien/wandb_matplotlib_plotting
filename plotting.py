from scipy.signal import savgol_filter
import re
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
import pprint 
marker_map = {
's_mup_final':'D',
's_sp_final':'D',
'm_sp_final':'D',
'm_mup_final':'D',
'muadam':'D',
'velo':'D',
# 'adam32':'D',
    'lion':'D',
    'adam':'D',
    'adamw':'D',
    'sgd':'D',
             
}
color_map = {
    
    's_mup_final':'grey',
    'm_mup_final':'black',

    's_sp_final':'orchid',
    'm_sp_final':'darkviolet',

    'muadam32_s':'royalblue',
    'muadam32_m':'darkblue',

    'velo':'darkorange',
    
    'adam32':'red',
    'muadam':'pink',
    
    'lion':'yellow',
    'adam':'brown',
    'adamw':'red',
    'sgd':'green',

}

label_map = {
    's_sp_final':'$LO_S$',
    'm_sp_final':'$LO_M$',

    's_mup_final':'$\mu LO_S$',
    'm_mup_final': '$\mu LO_M$',

    'muadam32_s':'$\mu$Adam$_S$',
    'muadam32_m':'$\mu$Adam$_M$',

    'velo':'VeLO',
    'muadam':'$\mu$Adam',
    
    'adam':'Adam',
    'adamw':'AdamW',
    'sgd':'SGD',
    'lion':'Lion',
            
}


linestyle_map = {
    's_sp_final':'-',
    'm_sp_final':'-',

    's_mup_final':'-',
    'm_mup_final': '-',

    'muadam32_s':'-',
    'muadam32_m':'-',

    'velo':'-',

    # 'adam32':'-',
        'muadam':'-',
    'adam':'-',
    'adamw':'-',
    'sgd':'-',
    'lion':'-',
}

order = [
    'lion',
    'adam',
    'adamw',
    'sgd',
    'muadam32_s',
    'muadam32_m',
    'velo',
    'muadam',
    's_sp_final',
    'm_sp_final',
    's_mup_final',
    'm_mup_final',

]
for n in order:
    for mp in [marker_map,color_map,label_map,linestyle_map]:
        if n not in mp:
            mp[n] = 'default'


def get_plt_attributes(lab,
                       idx,
                       run_info,
                       mapper=None,
                       marker_map=marker_map,
                       color_map=color_map,
                       label_map=label_map,
                       linestyle_map=linestyle_map,
                       return_k=False):
    mxl = 0
    k = None
    for key in marker_map.keys():
        if key in lab:
            if mxl < len(key):
                # print(lab,key)
                k = key
                mxl = len(key)
    # assert k != None, "Invalid label used: "+lab
    
    # if k == 'm_sp_final' and 'm_mup_final' in lab:
    #     k = 'm_mup_final'
        
    
    if 'muadam32' in lab and 'm_mup_final' in lab:
        k = 'muadam32_m'
        
    if 'muadam32' in lab and 's_mup_final' in lab:
        k = 'muadam32_s'
        
    if 'lion' in lab and 'm_sp_final' in lab:
        k = 'lion'
        
    if 'adam' in lab and 'm_sp_final' in lab:
        k = 'adam'
        
    if 'adamw' in lab and 'm_sp_final' in lab:
        k = 'adamw'
        
    if 'sgd' in lab and 'm_sp_final' in lab:
        k = 'sgd'
        
        
        
    if return_k:
        return k
    else:
        m,c,lab,ls = marker_map[k], color_map[k], label_map[k], linestyle_map[k]
        if mapper:
            c = mapper.to_rgba(idx)
        return m,c,lab,ls
    
    
    

def re_order(lab, order=order):
    assert len(set(lab)) == len(lab), "some labels are not unique" + lab
    
    lmap = {v:get_plt_attributes(v,idx=0,run_info={},return_k=True) for v in lab}
    out = []
    # print(lmap,order)
    for x in order:
        for k,v in lmap.items():
            if v == x:
                out.append(k)
    
    
    # assert len(out) == len(lab), "output labels and input label sizes dont match \nout:{}\nlab:{}".format(out,lab)
    return out

def exponential_moving_average(data, alpha=0.1):
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return ema

def rolling_average(data, window_size=4, sigma=2.0, alpha=0.1, polyorder=4, tpe='rolling'):
    """Compute a rolling average over a 1D array."""
    if tpe =='rolling':
        cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
        ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
        return ma_vec
    elif tpe == 'ema':
        return exponential_moving_average(data, alpha)
    elif tpe == 'savgol':
        return savgol_filter(data, window_length=window_size, polyorder=polyorder)
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
    # return savgol_filter(data, window_length=window_size, polyorder=2)
    
    
def get_speedup(loss_val,arr):
        idx = np.where(arr < loss_val)[0]
        if len(idx) == 0:
            return "--" #len(arr) + 1
        else:
            return "${}x$".format(round(1000/idx[0],2))
        
def get_speedup_std(loss_val,arr,arrt,arrb):
    idx = np.where(arr < loss_val)[0]
    idxt = np.where(arrt < loss_val)[0]
    idxb = np.where(arrt < loss_val)[0]

    b = arrb
    if len(idx) == 0:
        return "--" #len(arr) + 1
    else:
        mean = 1000/idx[0]
        if len(idxt) == 0 and len(idxb) == 0:
            return "${}x$".format(round(mean,2))
        elif len(idxt) == 0:
            std = np.abs(mean - 1000/idxb[0])
            return "${}\\pm{}$".format(round(mean,2),round(std,2))
        elif len(idxb) == 0:
            std = np.abs(mean - 1000/idxt[0])
            return "${}\\pm{}$".format(round(mean,2),round(std,2))
        else:
            std = np.mean([np.abs(mean - 1000/idxb[0]),np.abs(mean - 1000/idxb[0])])
            return "${}\\pm{}$".format(round(mean,2),round(std,2))


def get_iters_to_loss(loss_val,arr):
    idx = np.where(arr < loss_val)[0]
    if len(idx) == 0:
        return "--" #len(arr) + 1
    else:
        return "${}$".format(idx[0] + 1)
    

        
def get_parse_dict(k):
    run_string = k.split("['")[-1].split("']")[0]
    if not isinstance(run_string, str):
        raise TypeError("Expected a string as input")
    
    # Strip square brackets and single quotes if present
    if run_string.startswith("['") and run_string.endswith("']"):
        run_string = run_string[2:-2]
    
    # Define the regex pattern to match the string format
    pattern = r'(?P<architecture>\w+)-w(?P<width>\d+)-d(?P<depth>\d+)_(?P<dataset>\w+)-(?P<input_size>[\w\d\-x]+)'
    
    # Use re.match to find the pattern in the string
    match = re.match(pattern, run_string)
    
    if match:
        # Extract the matched groups into a dictionary
        run_info = match.groupdict()
        
        # Convert width and depth to integers
        run_info['width'] = int(run_info['width'])
        run_info['depth'] = int(run_info['depth'])
        
        return run_info
    else:
        raise ValueError(f"The input string {run_string} does not match the expected format")


    
run_string = "['transformer-w1024-d16_lm1b-s64-v32k']"
result = get_parse_dict(run_string)
print(result)  
    
    
def sort_values_by_criteria(d):
    """
    Sorts a list of dictionary keys based on specific criteria (WU, MaxLR, and It values) parsed from each key string.
    The sorting is stable, meaning elements with the same value for the sorting keys do not change position relative to each other.

    :param keys: List of keys in string format.
    :return: Sorted list of keys based on the specified criteria.
    """
    def parse_and_sort_criteria(key):
        """
        Parses a key string to extract sorting criteria and returns a tuple of these criteria for sorting.
        """
        parsed_dict = get_parse_dict(key)
        return (
            float(parsed_dict['width']),
            float(parsed_dict['depth']),
        )

    return sorted(d, key=parse_and_sort_criteria)



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
           mapper_key='width',
           reorder=True,
           ylog=False,
           def_cmap=plt.cm.plasma,
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
           filter_args=dict(tpe='none',window_size=2, sigma=4.0, alpha=0.1, polyorder=1,),
           show_meta_train_boundary=True):
    

    
    # assert last_slowmo_value != None, "no slowmo model added for comparison"
    
    speedup_to_slowmo = {}
    iters_to = {}
    
    # values = sort_values_by_criteria(values)
                                 
    if use_colormap:
        numerical_values = [float(get_parse_dict(k)[mapper_key]) for k in values]
        numerical_values = [x for x in range(len(values))]
        # print(numerical_values)
        
        # Choose a colormap
        colormap = def_cmap

        # Normalize your numerical values to [0, 1]
        if log_cmap:
            norm = mcolors.LogNorm(vmin=min(numerical_values)+1e-4, vmax=max(numerical_values))
        else:
            norm = mcolors.Normalize(vmin=min(numerical_values), vmax=max(numerical_values))

        # Create a ScalarMappable object with the chosen colormap and normalization
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    else:
        mapper = None
                                 
                                 
                                 
    
    if reorder:
        values = re_order(values, order=order)
        data = {k:data[k] for k in values}
        
        
    fig = plt.figure(figsize=figsize)
    
    
    ax = fig.add_subplot(1, 1, 1)
    if ylog:
        plt.yscale('log')
        

    eymin, eymax = [],[]
    for i, (y, df) in enumerate(data.items()):
        # run_info = get_parse_dict(y)

        # print(df)

        if verbose:
            print(i,y)
        m,c,lab,ls = ovr_legend[y]
                

        if lab_suffix is not None:
            suff = lab_suffix[i]
        else:
            suff = ''

        no_smooth = len(np.where(df['mean'][:20] >= 20)[0]) > 0

        
        df['mean'][np.where(df['mean'] >= 20)[0]] = 20
        df['stderr'][np.where(df['stderr'] >= 20)[0]] = 20

        
        yval = df['mean'][::skipfactor].astype(np.float32)
        # yval[np.where(yval > 1e6)[0]] = 1e6
        steps = df['steps'][::skipfactor].astype(np.float32)
        stderr = df['stderr'][::skipfactor].astype(np.float32)
        # stderr[np.where(stderr > 1e6)[0]] = 1e6
        max_vals = yval + stderr
        min_vals = yval - stderr

        
        if xlim is not None:
            a = np.where(steps > xlim[-1])[0]
            if len(a) != 0:
                steps = steps[:a[0]]
                yval = yval[:a[0]]
                max_vals = max_vals[:a[0]]
                min_vals = min_vals[:a[0]]
        # print("yval", yval.shape)

        yval_before = yval.shape[0]
        if smooth_mean and no_smooth == False :
            yval = rolling_average(yval, **filter_args)
            if yval_before != yval.shape[0]:
                steps = steps[:yval.shape[0]]
                stderr = stderr[:yval.shape[0]]
                max_vals = max_vals[:yval.shape[0]]
                min_vals = min_vals[:yval.shape[0]]
                
        # print("yval", yval.shape)
                    
        eymin.append(np.min(yval))
        eymax.append(yval[0])

        print("plotting ",lab+suff)
        plt.plot(steps, yval, label=lab+suff, color=c, linestyle=linestyle_ovrr.get(i, ls), linewidth=linewidth)

        
        
        if use_std:
            if smooth_std :
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
        if verbose:
            print()
                
            # speedup_to_slowmo[lab] = get_speedup_std(last_slowmo_value,yval,max_vals,min_vals)
            
    if sciy:
        # Set the formatter for the y-axis of this specific axis
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))  # Use scientific notation if outside this range

        ax.yaxis.set_major_formatter(formatter)

        # Optionally, add an offset text (scientific notation) to the axis
        ax.yaxis.get_offset_text().set_visible(True)
    
    # plt.xlim(xlim)
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
    
    # ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    # ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
            
    if show_meta_train_boundary:
        # t = plt.text(1050, 9.35, "Meta-train Horizon", verticalalignment='bottom', horizontalalignment='left', fontsize=12, rotation=0, color='darkred') 
        # t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='gray',linewidth=0))
        ax.axvline(x=1000, color='darkred', linestyle='--',linewidth=1.5) #,label="Meta-train Horizon")
        
    if use_legend:
        leg = plt.legend(fontsize=legend_fs,loc=legend_loc)
        
    
        # Adjust line width for each line in the legend
        for handle in leg.legend_handles:
            handle.set_linewidth(3.0)

        # leg.savefig("legend.pdf")

    
        
    def reject_outliers(data, m=2):
        if len(data) == 1:
            return data
        try:
            out = data[abs(data - np.mean(data)) < m * np.std(data)]
        except Exception:
            print("Exception reject_outliers data", data)
            raise Exception
        return out
    import math
    if autocenter:

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
        # print(eymin,eymax)

        #margin is 5% of graph
        margin = (eymax - eymin)  * 0.05
        #eymin-margin
        if bottom_zero:
            plt.ylim(0,eymax+margin)
        else:
            plt.ylim(eymin-margin,eymax+margin)
    
    plt.tight_layout()
    if savepath is not None:
        # os.makedirs(savepath,exist_ok=True)
        plt.savefig(savepath,
                    bbox_inches='tight')
        # plt.savefig(savepath.replace('.pdf','.png'),
        #             bbox_inches='tight')
    plt.show()
