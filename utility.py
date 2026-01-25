import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
import seaborn as sns


def castaing_integrand(ln_sigma, x, lambda2):
    '''
    Funzione usata per valutare la pdf di casting
    '''
    mu = -lambda2 
    sigma_val = np.exp(ln_sigma)
    
    # P_L: Gaussiana standard (media 0, dev std 1)
    p_l = (1 / (np.sqrt(2 * np.pi) * sigma_val)) * np.exp(-0.5 * (x / sigma_val)**2)
    
    # G: Kernel Gaussiano per ln(sigma) con varianza lambda2
    g_kernel = (1 / (np.sqrt(2 * np.pi * lambda2))) * np.exp(-0.5 * (ln_sigma - mu)**2 / lambda2)
    
    return p_l * g_kernel

def castaing_pdf(x, lambda2):
    '''
        Funzione per ottenere la pdf di casting
        x = dataset
        lambda2 = parametro
    '''
    # mi assicuro di avere un np array
    x = np.atleast_1d(x)
    
    # Definizione della griglia di integrazione per u = ln(sigma)

    sigma_u = np.sqrt(lambda2)
    mu = -lambda2  
    limit = 6.0 * sigma_u 
    
    u = np.linspace(mu - limit, mu + limit, 150)
    du = u[1] - u[0]
    u = u[None, :]   # Aggiungo asse per broadcasting
    
    # Calcolo dell'Integranda vettorializzato
    # L'integrale è: P(x) = Integral [ P_L(x/sigma) * (1/sigma) * G(u) ] du
    # Dove sigma = exp(u)    

    sigma_grid = np.exp(u)
    
    # Kernel Log-Normale G(u) (Dipende solo da u e lambda2)
    # G(u) = (1 / sqrt(2*pi*lambda2)) * exp( - (u - mu)^2 / (2*lambda2) )

    norm_G = 1.0 / (np.sqrt(2 * np.pi) * sigma_u)
    G_u = norm_G * np.exp(- (u - mu)**2 / (2 * lambda2))
    
    # Parte Gaussiana P_L(x/sigma) * (1/sigma)
    # x deve essere colonna (N x 1) per broadcastare contro u (1 x M) -> Matrice (N x M)
    X = x[:, None] 
    
    # P_L Normale Standard N(0,1)
    # P_L(z) = (1/sqrt(2pi)) * exp(-z^2/2) con z = x/sigma
    norm_P = 1.0 / np.sqrt(2 * np.pi)
    term_P = (norm_P / sigma_grid) * np.exp(- (X / sigma_grid)**2 / 2.0)
    
    # Integrazione (Regola dei Trapezi lungo l'asse u)
    # Moltiplichiamo i termini e integriamo
    integrand = term_P * G_u
    
    # Somma lungo l'asse 1 (l'asse di u) * passo du
    y_vals = np.trapezoid(integrand, dx=du, axis=1)
    
    if x.size == 1:
        return y_vals[0]
    return y_vals

def castaing_cdf(x, lambda2):
    """
    Calcola la CDF interpolando su una griglia fissa.
    """

    sigma_proxy = np.sqrt(lambda2) if lambda2 > 0 else 1.0
    limit = 100 * sigma_proxy  

    grid_x = np.linspace(-limit, limit, 5000)
    
    # Valuta la PDF sulla griglia (Vettorizzato)
    pdf_vals = castaing_pdf(grid_x, lambda2)
    
    # Integrazione Numerica Cumulativa
    cdf_grid = cumulative_trapezoid(pdf_vals, grid_x, initial=0)
    
    # ultimo 1.0
    cdf_grid /= cdf_grid[-1]
    
    # Interpolazione lineare
    return np.interp(x, grid_x, cdf_grid)

def normalize_ds(data):
    '''
        Normalizza i dati per il plot scale
    '''
    sigma = data.std()
    nbins = int(np.sqrt(len(data)))
    pdf, bin_edges = np.histogram(data-data.mean(), bins=nbins, density=True)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    x = bin_centers / sigma
    y =  pdf * sigma
    return(x,y)

def anderson_darling_statistic(data, cdf_function, args=()):
    """
    Calcola la statistica A^2 di Anderson-Darling per una distribuzione generica.
    
    Parameters:
    - data: array dei dati osservati (i tuoi incrementi)
    - cdf_function: la tua funzione CDF della Castaing
    - args: parametri della tua funzione
    """
    n = len(data)
    data = np.sort(data)
    
    # Calcola la CDF teorica per ogni punto dei dati osservati
    # u_i è la probabilità teorica di osservare quel dato
    u = cdf_function(data, *args)
    
    # per evitare log(0)
    u = np.clip(u, 1e-10, 1 - 1e-10)
    
    # 2. Formula di Anderson-Darling
    # S = sum( (2i - 1) * [ln(u_i) + ln(1 - u_{n+1-i})] )
    idx = np.arange(1, n + 1)
    S = np.sum((2 * idx - 1) * (np.log(u) + np.log(1 - u[::-1])))
    
    A2 = -n - S / n
    return A2

def rvs_castaing(lambda2, size=1):
    """
    Genera campioni casuali dalla distribuzione di Castaing.
    Logica:
    1. u ~ N(-lambda^2, lambda^2)  -> log-varianza
    2. sigma = exp(u)              -> deviazione standard locale
    3. x ~ N(0, sigma^2)           -> incremento finale
    """
    mu_u = -lambda2
    sigma_u = np.sqrt(lambda2)
    
    # Campioniamo le varianze logaritmiche
    u = np.random.normal(loc=mu_u, scale=sigma_u, size=size)
    sigmas = np.exp(u)
    
    # Campioniamo x condizionato a sigma (x = sigma * Gaussiana(0,1))
    z = np.random.normal(loc=0, scale=1, size=size)
    x = sigmas * z
    return x

def bootstrap_ad_test(observed_A2, best_lambda, n_samples_data, n_simulations=100):
    """
    FUNZIONE NON USATA, ABBIAMO TENUTO SOLO LA PARTE CON SUBSAMPLING    
    
    Esegue il test Monte Carlo per calcolare il p-value.
    
    Args:
    - observed_A2: il valore A^2 calcolato sui tuoi dati veri
    - best_lambda: il lambda stimato sui tuoi dati veri
    - n_samples_data: lunghezza del tuo dataset (len(data_s))
    - n_simulations: quanti dataset sintetici generare (consigliato >= 100)
    """
    sim_A2_scores = []
       
    for i in range(n_simulations):
        # Generiamo dati sintetici con il lambda osservato
        synthetic_data = rvs_castaing(best_lambda, size=n_samples_data)
        # Normalizzo
        synth_mean = np.mean(synthetic_data)
        synth_std = np.std(synthetic_data)
        synthetic_data_norm = (synthetic_data - synth_mean) / synth_std
        
        # Ottengo l'hist
        bin_centers_s, pdf_s = normalize_ds(synthetic_data) 
        
        try:
            # Fit
            p0 = [best_lambda]
            bounds = (0.001, 2.0)
            popt, _ = curve_fit(castaing_pdf, bin_centers_s, pdf_s, p0=p0, bounds=bounds)
            sim_lambda = popt[0]
            
            # Calcolo A^2 sintetico con il lambda appena stimato
            sim_score = anderson_darling_statistic(synthetic_data_norm, castaing_cdf, args=(sim_lambda,))
            sim_A2_scores.append(sim_score)
            
        except Exception as e:
            continue # Se il fit fallisce, saltiamo l'iterazione
            
    sim_A2_scores = np.array(sim_A2_scores)
    
    # Calcolo P-value: Frazione di score simulati maggiori di quello osservato
    p_value = np.mean(sim_A2_scores > observed_A2)
    
    return p_value, sim_A2_scores

def bootstrap_ad_test_subsampled(data_real, best_lambda, n_subsample=500, n_simulations=100):
    """
    Esegue il test AD su sottocampioni dei dati per evitare che N troppo grande
    faccia fallire il test per deviazioni trascurabili.
    """
    # Prendo un campione di lunghezza scelta dai miei dati

    subset_indices = np.random.choice(len(data_real), n_subsample, replace=False)
    data_subset = data_real[subset_indices]
    
    # Normalizzazione del sottocampione reale
    data_subset = (data_subset - np.mean(data_subset)) / np.std(data_subset)
    
    # Score osservato (sul subset)
    observed_A2 = anderson_darling_statistic(data_subset, castaing_cdf, args=(best_lambda,))
    
    sim_A2_scores = []
    
    for i in range(n_simulations):
        # Genero sottocampione sintetico
        synthetic_data = rvs_castaing(best_lambda, size=n_subsample)
        # Normalizzo
        syn_norm = (synthetic_data - np.mean(synthetic_data)) / np.std(synthetic_data)
        
        #uso per velocità del ciclo lo stesso lambda da cui sono estratti dati
        sim_score = anderson_darling_statistic(syn_norm, castaing_cdf, args=(best_lambda,))
        sim_A2_scores.append(sim_score)
    
    sim_A2_scores = np.array(sim_A2_scores)
    
    # P-value
    p_value = np.mean(sim_A2_scores > observed_A2)
    
    return p_value, observed_A2, np.mean(sim_A2_scores)

def get_res(df, Anderson = True,suppress_print=False):
    # Dizionari per salvare i risultati
    results = {} # bin_centers e pdf per ogni s
    lambdas = [] # valori di lambda2
    lambda_errors = [] # errori
    anderson_scores = {} # Dizionario per gli score AD
    for s in df.columns:

        data_s = df[s].dropna().values

        # PDF sperimentale
        bin_centers, pdf = normalize_ds(data_s)
        
        # Parametri iniziali e bounds
        p0 = [0.1]
        bounds = (0.001, 2.0)
        
        try:
            # fit
            popt, pcov = curve_fit(castaing_pdf, bin_centers, pdf, p0=p0, bounds=bounds)
            
            l2 = popt[0]
            l2_err = np.sqrt(np.diag(pcov))[0]
            
            # Salvataggio risultati
            lambdas.append(l2)
            lambda_errors.append(l2_err)
            results[s] = (bin_centers, pdf)

            if suppress_print is not True:
                print(f"Scale s={s}: Lambda^2 = {l2:.3f} +/- {l2_err:.3f}")
            
            # aggiungo lo score di anderson
            if Anderson:

                # Parte non usata dato che abbiamo effettuato subsampling per valutare i fit

                #sigma_s = np.std(data_s)
                #mean_s = np.mean(data_s)
                #data = (data_s - mean_s) / sigma_s
                #score = anderson_darling_statistic(data, castaing_cdf, args=(l2,))
                #anderson_scores[s] = score
                #print(f"   -> Anderson Score: {score:.4f}")
                #p_val, sim_scores = bootstrap_ad_test(score, l2, len(data_s), n_simulations=200)
                
                data_clean = data_s[~np.isnan(data_s)]

                p_val, obs_a2, mean_sim_a2 = bootstrap_ad_test_subsampled(
                    data_clean, 
                    l2,
                    n_subsample=500, 
                    n_simulations=100
                )
                if suppress_print is not True:
                    print(f"Scale {s}: P-value={p_val:.3f} (Obs A2={obs_a2:.2f}, Mean Sim A2={mean_sim_a2:.2f})")
                
        except Exception as e:
            print(f"Fitting error for scale {s}: {e}")

    return(results,lambdas,lambda_errors,anderson_scores)

def plot_results(results, lambdas, lambda_errors, show_collapse=True, show_lambda=True):
    """
    Plotta i risultati dell'analisi di invarianza di scala.
    
    Args:
        results: Dizionario con i dati per ogni scala.
        lambdas: Lista dei valori lambda^2.
        lambda_errors: Lista degli errori su lambda^2.
        show_collapse (bool): Se True, mostra il Collapse Plot (sinistra).
        show_lambda (bool): Se True, mostra il grafico Lambda vs Log(s) (destra).
    """
    
    # Media Pesata
    weights = 1 / np.array(lambda_errors)**2
    lambda_mean = np.average(lambdas, weights=weights)
    err_mean = 1 / np.sqrt(np.sum(weights))

    print(f"W. Mean Lambda^2 +/- err: {lambda_mean:.4f} +/- {err_mean:.4f}")

    # Se non è richiesto nessun plot, esci
    if not show_collapse and not show_lambda:
        return

    # 2. Configurazione dinamica della figura
    if show_collapse and show_lambda:
        fig, (ax1, ax2) = plt.subplots(figsize=(16, 6), nrows=1, ncols=2)
    elif show_collapse:
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax2 = None
    elif show_lambda:
        fig, ax2 = plt.subplots(figsize=(8, 6))
        ax1 = None

    # A: Collapse Plot (Scale Invariance)
    if show_collapse and ax1 is not None:
        colors = plt.colormaps['viridis'](np.linspace(0, 1, len(results)))

        for i, s in enumerate(results):
            bc, p = results[s]
            ax1.semilogy(bc, p, 'o', markersize=3, alpha=0.5, color=colors[i], label=f's={s}')

        # Curva teorica
        x_theory = np.linspace(-5, 5, 100)
        y_theory = castaing_pdf(x_theory, lambda_mean)
        
        ax1.semilogy(x_theory, y_theory, 'r-', linewidth=2.5, label=rf'Global Fit $\lambda^2={lambda_mean:.2f}$')
        ax1.set_title('Collapse Plot: Critical Scale Invariance')
        ax1.set_xlabel(r'$\Delta_s B / \sigma$')
        ax1.set_ylabel(r'$\sigma_s \cdot P(\Delta_s B)$')
        ax1.set_ylim(5*1e-4, 2)
        
        
        if len(results) > 10:
            # Mostra solo fit globale nella legenda se ci sono troppe voci
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend([handles[-1]], [labels[-1]], loc='best')
        else:
            ax1.legend(loc='best', fontsize='small')
            
        ax1.grid(True, which="both", ls="-", alpha=0.3)

    # B: Lambda vs Log(s)
    if show_lambda and ax2 is not None:
        s_values = list(results)
        
        ax2.errorbar(s_values, lambdas, yerr=lambda_errors, fmt='ko', ls='--', capsize=3, label=r"$\lambda^2$ values")
        
        ax2.hlines(lambda_mean, min(s_values)/1.5, max(s_values)*1.5, colors='red', linestyles='-', label=r"Weighted Mean $\lambda^2$")
        
        ax2.fill_between([min(s_values)/1.5, max(s_values)*1.5], 
                         lambda_mean - err_mean, lambda_mean + err_mean, 
                         color='red', alpha=0.1, label=r"Mean Uncertainty ($1\sigma$)")

        ax2.set_xscale('log')
        ax2.grid(True, which="both", ls="-", alpha=0.3)
        ax2.set_xlabel(r"log(s)")
        ax2.set_ylabel(r"$\lambda^2$")
        ax2.set_title(rf"Dependence of $\lambda^2$ on Scale $s$")
        ax2.set_xlim(min(s_values)/1.5, max(s_values)*1.5)
        ax2.legend(loc='best')

    plt.tight_layout()
    plt.show()

def lambda_comp(lambdas,lambda_errors,s):
    '''
        Valuta la compatibilità dei lambda^2 ottenuti con una retta a pendenza nulla
    '''
    def line(x,m,b):
        return(m*x+b)
    popt,pcov = curve_fit(line,np.log(s),lambdas,[0,0.16],lambda_errors)
    m = popt[0]
    dm = np.sqrt(np.diag(pcov))[0]
    print(f"Fitting a line to Lambda^2, we get y=m log(s)+b with:\nm +/- dm = {m:.3f} +/- {dm:.3f}")

def plot_collapse(increments_df):
    """
    increments_df è un dataframe dove le colonne sono i diversi valori di s
    """
    plt.figure(figsize=(10, 6))
    
    def plot_single_column(col):
        s = col.name
       
        # Rimozione eventuali NaN (non dovrebbero essercene)
        data = col.dropna()

        x,y = normalize_ds(data)
        # Plot diretto
        plt.semilogy( x , y , ls='', marker = 'o',label=f's={s}',ms = 3)

    # Applichiamo la funzione a ogni colonna (axis=0 è il default)
    increments_df.apply(plot_single_column)

    plt.title('Data Collapse: Rescaled PDFs')
    plt.xlabel(r'$\Delta_s B / \sigma_s$')
    plt.ylabel(r'$\sigma_s \cdot P(\Delta_s B)$')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()


def standardize_increments(increments:np.ndarray):
    increments_standardized=(increments-np.mean(increments))/np.std(increments)
    return increments_standardized

def histo_kde(increments:np.ndarray, h_ax, bin_edges, scale, xlabel="Standardized increments", hist_color='lightblue', line_color='blue'):
    # assuming increments array (at a fixed scale),
    # and a single "heart" sample;
    # NB requires existing ax
    h_ax.set_xlabel(xlabel)
    h_ax.set_ylabel("Frequence")
    h_ax.hist(increments, bins=bin_edges, density=True, color=hist_color, label='Data')
    sns.kdeplot(increments, color=line_color, linewidth=2, label="KDE-estimated PDF", ax=h_ax)
    h_ax.legend()
    h_ax.set_title(f"Increments (PDF) at s={scale}")
    #plt.savefig('./point4_histo.png', dpi=300, bbox_inches='tight')

def gaussian_vs_cast_fit(increments:np.ndarray, g_ax, bin_edges, lambda2, scale, scatter_color='lightblue', scatter_marker='o', line_color='r'):
    # increments should be already standardized here
    # NB requires existing ax
    g_ax.set_xlabel("Standardized increments")
    g_ax.set_ylabel("Frequence")
    
    heights, edges = np.histogram(increments, bins=bin_edges, density=True) # NB density=True to get normalized histogram
    centers = (edges[:-1] + edges[1:]) / 2
    g_ax.scatter(centers, heights, color=scatter_color, marker=scatter_marker, label='Data')
    
    mu, sigma = sp.stats.norm.fit(increments)
    lin = np.linspace(increments.min(), increments.max(), 1000)
    g_ax.plot(lin, sp.stats.norm.pdf(x=lin, loc=mu, scale=sigma), '--', color='red', label='Gaussian fit')
    g_ax.plot(lin, castaing_pdf(lin, lambda2), '-', color='blue', label='Castaing fit')
    g_ax.set_yscale('log')
    g_ax.set_ylim(bottom=10**(-3.125))
    g_ax.legend()
    g_ax.grid(True, alpha=0.3)
    g_ax.set_title(f"Gaussian vs. Castaing fit of increments at s={scale}")
    #plt.savefig('./point4_gauss_vs_cast_fit.png', dpi=300, bbox_inches='tight')
    print(f"Gaussian vs Castaing fit (mean {mu:.4f}, standard deviation {sigma:.4f})")
    return mu, sigma

def ks_tests(increments, lambda2, cdf_ax, scale):
    # Kolmogorov-Smirnov (KS) test of Gaussian vs. Castaing fits to data.
    # essentially compares the two CDFs
    # NB requires existing ax
    
    print("\n=== Kolmogorov-Smirnov tests ===")
    ks_stat_norm, p_val_norm = sp.stats.kstest(
        increments, 
        lambda x: sp.stats.norm.cdf(x, 0, 1)
    )
    ks_stat_cast, p_val_cast = sp.stats.kstest(
        increments, 
        lambda x: castaing_cdf(x, lambda2)
    )
    print(f"Castaing: KS={ks_stat_cast:.4f}")
    print(f"Normal:   KS={ks_stat_norm:.4f}")
    
    # plot: empirical vs theoretical CDFs (NB have to sort increments first)
    increments_sorted = np.sort(increments)
    empirical_cdf = np.arange(1, len(increments_sorted) + 1) / len(increments_sorted)
    
    castaing_theoretical = castaing_cdf(increments_sorted, lambda2)
    normal_theoretical = sp.stats.norm.cdf(increments_sorted, loc=0, scale=1)
    
    cdf_ax.plot(increments_sorted, empirical_cdf, 'k-', linewidth=2, label='Empirical')
    cdf_ax.plot(increments_sorted, castaing_theoretical, 'r--', linewidth=2, 
             label=f'Castaing (λ²={lambda2}, KS={ks_stat_cast:.3f})')
    cdf_ax.plot(increments_sorted, normal_theoretical, 'b--', linewidth=2, 
             label=f'Normal (KS={ks_stat_norm:.3f})')
    cdf_ax.set_xlabel('Standardized increments')
    cdf_ax.set_ylabel('Cumulative distribution function')
    cdf_ax.set_title(f'CDF Comparison at s={scale}')
    cdf_ax.legend()
    cdf_ax.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.savefig('./point4_cdf_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()