import math
import numpy as np

F_MAINS = 60 # mains frequency in Hz
POINTS_PER_CYCLE = 500 # number of samples per cycle
DT = 1 / (F_MAINS * POINTS_PER_CYCLE) # time step

class CurrentDecomposition:
    def __init__(self, i_active, i_reactive, i_void):
        self.i_active = i_active
        self.i_reactive = i_reactive
        self.i_void = i_void

# Simple Moving Average Filter
class sMAF:
    def __init__(self):
        self.MAF_input = 0.0
        self.MAF_KONTz = 0
        self.vector_jm = [0.0] * POINTS_PER_CYCLE
        self.MAF_summedz = 0.0
        self.mean_out = 0.0
        self.inicio = 0

# RMS calculation
class RMS:
    def __init__(self):
        self.inicio = 0
        self.NA = 0
        self.temp = 0.0
        self.jm = [0.0] * POINTS_PER_CYCLE
        self.summed = 0.0
        self.v_RMS = 0.0

# Unbiased Integral
class UI:
    def __init__(self):
        self.integral = 0.0
        self.integral_old = 0.0
        self.out = 0.0
        self.inicio = 0

# Moving Average Filter function
def maf(m_a_f, m_input, n_samples):
    if m_a_f.inicio == 0:
        m_a_f.MAF_KONTz = 0
        m_a_f.vector_jm = [0.0] * n_samples
        m_a_f.MAF_summedz = 0
        m_a_f.mean_out = 0
        m_a_f.inicio = 1

    if m_a_f.MAF_KONTz == n_samples:
        m_a_f.MAF_KONTz = 0

    m_a_f.MAF_input = m_input
    m_a_f.MAF_summedz += m_a_f.MAF_input - m_a_f.vector_jm[m_a_f.MAF_KONTz]
    m_a_f.vector_jm[m_a_f.MAF_KONTz] = m_a_f.MAF_input
    m_a_f.MAF_KONTz += 1
    m_a_f.mean_out = m_a_f.MAF_summedz / n_samples

# RMS calculation function
def rms(r_m_s, RMS_input, n_samples):
    if r_m_s.inicio == 0:
        r_m_s.temp = 0
        r_m_s.NA = 0
        r_m_s.jm = [0.0] * n_samples
        r_m_s.summed = 0
        r_m_s.inicio = 1

    r_m_s.temp = RMS_input
    r_m_s.summed = r_m_s.summed + r_m_s.temp - r_m_s.jm[r_m_s.NA]
    r_m_s.jm[r_m_s.NA] = r_m_s.temp
    r_m_s.NA += 1

    if r_m_s.NA >= n_samples:
        r_m_s.NA = 0

    if r_m_s.summed <= 0:
        r_m_s.summed = 0.0001

    r_m_s.v_RMS = math.sqrt(r_m_s.summed / n_samples)

# Unbiased Integral function
def ui(ui_obj, av, ui_input):
    if ui_obj.inicio == 0:
        ui_obj.integral = 0
        ui_obj.integral_old = 0
        ui_obj.out = 0
        ui_obj.inicio = 1

    ui_obj.integral += (DT / 2) * (ui_input + ui_obj.integral_old)
    ui_obj.integral_old = ui_input
    maf(av, ui_obj.integral, POINTS_PER_CYCLE)
    ui_obj.out = ui_obj.integral - av.mean_out

# CPT function for single-phase
def CPT(Vs, Is): 
    print("Initiating CPT decomposition...")
    for x in range(0 , len(Vs[0])): # iterate over samples
        u = Vs[0][x]
        i = Is[0][x]

        if x == 0:
            v_u, v_i = [], [] # voltage and current vectors
            v_ia, v_ir, v_iv = [], [], [] # active, reactive, and void current vectors

            v_iv_rms = []  # List to hold RMS values for each window

            ui_u = UI() # unbiased integral for voltage
            mui_u = sMAF() # moving average filter for unbiased integral
            UI_u = RMS() 
            U, I = RMS(), RMS() # RMS for voltage and current
            P = sMAF()
            W = sMAF()

        v_u.append(u)
        v_i.append(i)

        ui(ui_u, mui_u, u)
        rms(U, u * u, POINTS_PER_CYCLE)
        rms(I, i * i, POINTS_PER_CYCLE)
        rms(UI_u, ui_u.out * ui_u.out, POINTS_PER_CYCLE)
        maf(P, u * i, POINTS_PER_CYCLE)
        P.mean_out = max(P.mean_out, 0.000001)
        maf(W, ui_u.out * i, POINTS_PER_CYCLE)

        # Active current
        A = (P.mean_out / (U.v_RMS ** 2)) * u if U.v_RMS != 0 else 0
        v_ia.append(A)

        # Reactive current
        R = (W.mean_out / (UI_u.v_RMS ** 2)) * ui_u.out if UI_u.v_RMS != 0 else 0
        v_ir.append(R)

        # Void current
        V = i - A - R
        v_iv.append(V)

        i_void_rms = math.sqrt(max(I.v_RMS**2 - (P.mean_out / U.v_RMS)**2 - (W.mean_out / UI_u.v_RMS)**2, 0))
        v_iv_rms.append(i_void_rms) 

    def steady_removal(signal, threshold=0.01):
        window_size = POINTS_PER_CYCLE

        lenSignal = len(signal)
        if lenSignal < 2 * window_size: # Not enough data to process
            return signal

        num_windows = lenSignal // window_size
        if num_windows < 2: # Not enough windows to analyze
            return signal

        i_void_rms = np.array(v_iv_rms)

        windows = []
        
        for window_index in range(num_windows):
            # Pegar os samples desta janela
            start_idx = window_index * window_size
            end_idx = (window_index + 1) * window_size
            window_samples = i_void_rms[start_idx:end_idx]
            
            window_mean = np.mean(window_samples)
            windows.append(window_mean)

        diffs = np.abs(np.diff(windows))

        if len(diffs) == 0 or max(diffs) < threshold: # No significant change detected
            idx_cut = 2 * window_size
        else:
            idx_cut = 2 * window_size * (1 + np.argmax(diffs))

        if idx_cut >= lenSignal: # If cut index exceeds signal length, return original signal
            return signal
        
        return signal[idx_cut:]
        
    i_active = steady_removal(np.array(v_ia))
    i_reactive = steady_removal(np.array(v_ir))
    i_void = steady_removal(np.array(v_iv))

    print("Decomposition CPT completed.")
    return CurrentDecomposition(i_active, i_reactive, i_void)