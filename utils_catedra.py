import numpy as np
import matplotlib.pyplot as plt

def rcos(beta, span, sps, shape='sqrt'):
    """Generate a raised cosine or root raised cosine filter.

    This function replicates the MATLAB `rcosdesign()` function, generating the impulse response
    of a raised cosine or root raised cosine filter used in digital communications for
    pulse shaping.

    Parameters
    ----------
    beta : float
        Roll-off factor, must be between 0 and 1.
    span : int
        Number of symbols. The filter length will be span * sps + 1.
    sps : int
        Samples per symbol.
    shape : str, optional
        Filter shape, either 'normal' for raised cosine or 'sqrt' for root raised cosine.
        Default is 'sqrt'.

    Returns
    -------
    numpy.ndarray
        The filter impulse response, normalized to unit energy.

    Raises
    ------
    ValueError
        If beta is not in [0, 1] or shape is not 'normal' or 'sqrt'.

    Notes
    -----
    The filter is normalized such that the sum of squares of the coefficients is 1.
    For beta=0, it reduces to a `sinc()` function.

    Examples
    --------
    >>> import numpy as np
    >>> h = rcos(0.5, 6, 64, 'sqrt')
    >>> h.shape
    (49,)
    """
    if not (0 <= beta <= 1):
        raise ValueError("beta debe estar en [0, 1]")
    if shape.lower() not in ('sqrt', 'normal'):
        raise ValueError("shape debe ser 'sqrt' o 'normal'")
    
    N = span * sps
    t = np.linspace(-span/2, span/2, N+1)
    
    if beta == 0:
        p = np.sinc(t)
    
    elif shape.lower() == 'normal':
        sinc_t = np.sinc(t)
        cos_term = np.cos(np.pi * beta * t)
        den = 1 - (2 * beta * t)**2
        p = np.divide(sinc_t * cos_term, den, out=np.zeros_like(den), where=den != 0)
        
        # Handle the singularity at t = 1/(2*beta)
        special_mask = np.abs(den) < 1e-8
        if np.any(special_mask):
            p[special_mask] = (np.pi / 4) * np.sinc(1 / (2 * beta))
    
    else:  # sqrt
        t_abs = np.abs(t)
        p = np.zeros_like(t)
        
        # Special case at t=0
        mask_zero = t_abs < 1e-8
        p[mask_zero] = (1 - beta) + 4 * beta / np.pi
        
        # Special case at t = 1/(4*beta)
        special_t_sqrt = 1 / (4 * beta)
        mask_special = np.abs(t_abs - special_t_sqrt) < 1e-8
        if np.any(mask_special):
            p[mask_special] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        
        # General case
        mask_general = ~mask_zero & ~mask_special
        if np.any(mask_general):
            ti = t[mask_general]
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti)**2)
            p[mask_general] = num / den
    
    # Normalización de energía
    p = p / np.sqrt(np.sum(p**2))
    return p

def upfir(h, x, up=1):
    """
    Replica la función upfirdn de MATLAB (solo upsample y FIR).
    h: respuesta al impulso del filtro
    x: señal de entrada
    up: factor de upsampling
    """
    # Upsample
    xu = np.zeros(len(x) * up)
    xu[up//2::up] = x
    # Filtrado
    y = np.convolve(xu, h, mode='same')
    return y

def eyediagram(Y_signal, sps, n_traces=None, cmap='viridis', 
             N_grid_bins=350, grid_sigma=3, ax=None, **plot_kw):
    """Plots a colored eye diagram, internally calculating color density.

    Parameters
    ----------
    Y_signal : np.ndarray
        Full amplitude array (1D).
    sps : int
        Samples per symbol. Used to segment the eye traces.
    n_traces : int, optional
        Maximum number of traces to plot. If None, all available traces
        will be plotted. Defaults to None.
    cmap : str, optional
        Name of the matplotlib colormap. Defaults to 'viridis'.
    N_grid_bins : int, optional
        Number of bins for the density histogram. Defaults to 350.
    grid_sigma : float, optional
        Sigma for the Gaussian filter applied to the density. Defaults to 3.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure and axes.
        Defaults to None.
    **plot_kw : dict, optional
        Additional plotting parameters:
        
        Figure parameters (used only if ax is None):
        - figsize : tuple, default (10, 6)
        - dpi : int, default 100
        
        Line collection parameters:
        - linewidth : float, default 0.75
        - alpha : float, default 0.25
        - capstyle : str, default 'round'
        - joinstyle : str, default 'round'
        
        Axes formatting parameters:
        - xlabel : str, default "Time (2-symbol segment)"
        - ylabel : str, default "Amplitude"
        - title : str, default "Eye Diagram ({num_traces} traces)"
        - grid : bool, default True
        - grid_alpha : float, default 0.3
        - xlim : tuple, optional (xmin, xmax)
        - ylim : tuple, optional (ymin, ymax)
        - tight_layout : bool, default True
        
        Display parameters:
        - show : bool, default True (whether to call plt.show())
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the eye diagram plot.
    """
    # Truncate signals to avoid edge artifacts
    # Remove sps//2 samples from each end
    start_idx = sps // 2
    end_idx = len(Y_signal) - sps // 2
    
    if start_idx >= end_idx or end_idx <= start_idx:
        raise ValueError(f"Signal too short for truncation. Need at least {sps} samples, got {len(Y_signal)}.")
    
    Y_truncated = Y_signal[start_idx:end_idx]
    
    # Eye diagram parameters
    num_points_per_trace = 2 * sps
    if len(Y_truncated) < num_points_per_trace:
        raise ValueError(f"Need at least {num_points_per_trace} points for eye diagram, got {len(Y_truncated)} after truncation.")
    
    available_traces = len(Y_truncated) // num_points_per_trace
    num_traces_to_plot = min(available_traces, n_traces) if n_traces is not None else available_traces
    
    if num_traces_to_plot == 0:
        raise ValueError(f"Not enough points to form even one trace of {num_points_per_trace} points after truncation.")
    
    X_truncated = np.kron(np.ones(num_traces_to_plot), np.linspace(-1, 1 - 1/sps, num_points_per_trace))
    Y_truncated = Y_truncated[:num_traces_to_plot * num_points_per_trace]
    
    # Get colormap
    try:
        cmap_obj = getattr(plt.cm, cmap)
    except AttributeError:
        print(f"Warning: Colormap '{cmap}' not found. Using 'viridis' by default.")
        cmap_obj = plt.cm.viridis

    # Extract plotting parameters with defaults
    # Figure parameters
    figsize = plot_kw.get('figsize', None)
    dpi = plot_kw.get('dpi', 100)
    
    # Line collection parameters
    linewidth = plot_kw.get('linewidth', 1)
    alpha = plot_kw.get('alpha', 0.05)
    capstyle = plot_kw.get('capstyle', 'round')
    joinstyle = plot_kw.get('joinstyle', 'round')
    
    # Axes formatting parameters
    xlabel = plot_kw.get('xlabel', "Time (2-symbol segment)")
    ylabel = plot_kw.get('ylabel', "Amplitude")
    title_template = plot_kw.get('title', "Eye Diagram ({num_traces} traces)")
    grid = plot_kw.get('grid', True)
    grid_alpha = plot_kw.get('grid_alpha', 0.3)
    xlim = plot_kw.get('xlim', None)
    ylim = plot_kw.get('ylim', None)
    tight_layout = plot_kw.get('tight_layout', True)
    
    # Display parameters
    show = plot_kw.get('show', True)

    # Calculate ranges using truncated signals
    min_x, max_x = X_truncated.min(), X_truncated.max()
    min_y, max_y = Y_truncated.min(), Y_truncated.max()
    
    # Normalize coordinates (handle edge cases)
    X_norm = np.zeros_like(X_truncated) if max_x == min_x else (X_truncated - min_x) / (max_x - min_x)
    Y_norm = np.zeros_like(Y_truncated) if max_y == min_y else (Y_truncated - min_y) / (max_y - min_y)

    # Create density grid using truncated signals
    from scipy.ndimage import gaussian_filter
    grid_density, _, _ = np.histogram2d(X_truncated, Y_truncated, bins=N_grid_bins)
    grid_density = gaussian_filter(grid_density, sigma=grid_sigma)

    # Map points to grid indices
    ix_grid = np.clip((X_norm * (N_grid_bins - 1)).astype(int), 0, N_grid_bins - 1)
    iy_grid = np.clip((Y_norm * (N_grid_bins - 1)).astype(int), 0, N_grid_bins - 1)

    # Get and normalize color values
    color_values = grid_density[ix_grid, iy_grid]
    color_range = color_values.max() - color_values.min()
    color_values_norm = np.zeros_like(color_values) if color_range == 0 else (color_values - color_values.min()) / color_range
    

    # Prepare data for plotting using truncated signals
    x_eye_trace = X_truncated[:num_points_per_trace]

    Y_reshaped = Y_truncated.reshape(num_traces_to_plot, num_points_per_trace)
    color_reshaped = color_values_norm.reshape(num_traces_to_plot, num_points_per_trace)

    # Create plot or use existing axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        created_fig = True
    else:
        created_fig = False

    # Plot eye traces
    from matplotlib.collections import LineCollection
    for i in range(num_traces_to_plot):
        # Create line segments
        points = np.array([x_eye_trace, Y_reshaped[i]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        if len(segments) > 0:
            colors = cmap_obj(color_reshaped[i][:len(segments)])
            lc = LineCollection(segments, colors=colors, linewidth=linewidth, 
                              alpha=alpha, capstyle=capstyle, joinstyle=joinstyle)
            ax.add_collection(lc)

    # Format plot
    ax_ymin, ax_ymax = ax.get_ylim()
    min_y = np.minimum(min_y, ax_ymin)
    max_y = np.maximum(max_y, ax_ymax)
    ax.set_xlim(xlim if xlim is not None else (-1,1))
    ax.set_ylim(ylim if ylim is not None else (min_y, max_y))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Format title with number of traces
    title_formatted = title_template.format(num_traces=num_traces_to_plot)
    ax.set_title(title_formatted)
    
    if grid:
        ax.grid(True, alpha=grid_alpha)
    
    if created_fig and tight_layout:
        plt.tight_layout()
    
    if created_fig and show:
        plt.show()
    
    return ax


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Generate a sample signal (e.g., BPSK with noise)
    np.random.seed(0)
    num_symbols = 4000
    sps = 64  # samples per symbol
    symbols = np.random.choice([-1, 1], size=num_symbols)
    h = rcos(0.8, 22, sps, 'normal') # Raised cosine filter

    s = upfir(h, symbols, sps) # signal
    z = 0.01 * np.random.randn(num_symbols * sps) # noise
    
    r = s + z  

    # Plot the eye diagram
    eyediagram(r, sps, cmap='inferno')